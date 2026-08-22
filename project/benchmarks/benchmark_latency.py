"""
scripts/benchmark_latency.py

Latency benchmark for the HHGOa-2nd pipeline (retrieval-first voice/text
assistant). Drop this file into the `scripts/` folder next to
`retrieval_100_questions.py` and run it against a *running* backend
(`uvicorn project.api.app:app`).

WHAT IT MEASURES
-----------------
Two things per request:

1. `client_wall_ms`  — wall-clock time measured on the benchmark client,
   from just before the HTTP request is sent to just after the response
   is fully received. This is the number that matters for a P50/P70/P100
   latency submission, because it includes network + serialization
   overhead exactly the way a real caller would experience it.

2. Server-reported stage timings — pulled straight out of the
   `timings` block the API already returns (see project/api/text.py and
   project/timing/stage_timer.py), so you also get a P50/P70/P100
   breakdown per stage (guardrails, translation, retrieval, Groq, TTS,
   etc.) to see where the time actually goes.

WHY MULTIPLE QUERIES, NOT ONE
------------------------------
A single best-case run tells you almost nothing about tail latency.
This script runs a *set* of distinct questions, each repeated several
times (with a shuffle + warm-up), so the P70/P100 numbers reflect
real variance (cold caches, occasional slow Groq/Sarvam calls, etc.)
rather than one lucky sample.

USAGE
-----
    # Benchmark the multilingual text pipeline (POST /v1/text)
    python scripts/benchmark_latency.py --endpoint text --repeats 3

    # Benchmark the English query+TTS pipeline (POST /v1/query)
    python scripts/benchmark_latency.py --endpoint query --repeats 3

    # Point at a non-default host, save raw results, skip plots
    python scripts/benchmark_latency.py --base-url http://localhost:8000 \
        --repeats 5 --out results/latency_run1 --no-plot

Requires: requests (already in requirements.txt). matplotlib is optional
(only needed for --plot, which is on by default).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


QUESTIONS: List[str] = [
    "Why do tectonic plates move?",
    "What causes a rainbow?",
    "What causes tides in the ocean?",
    "What causes muscle soreness after exercise?",
    "What causes soil erosion?",
    "What causes air pollution?",
    "What is the difference between speed and velocity?",
    "What is potential energy?",
    "What does a DNS server do?",
    "How does cloud computing work?",
]

MULTILINGUAL_QUERIES: List[Dict[str, str]] = [
    {"text": "भूकंप क्यों आते हैं?", "language_code": "hi-IN"},
    {"text": "गुरुत्वाकर्षण क्या है?", "language_code": "hi-IN"},
]


# ============================================================
# ENDPOINT CONFIG
# ============================================================
# Each endpoint returns its timings block under a slightly different
# shape (see project/api/text.py vs project/api/app.py::query), so we
# normalize here rather than hardcoding one shape.

ENDPOINTS = {
    "text": {
        "path": "/v1/text",
        "total_key": "total_ms",
        "stage_keys": [
            "guardrail_ms",
            "translation_to_english_ms",
            "retrieval_wall_ms",
            "embedding_ms",
            "qdrant_ms",
            "fusion_ms",
            "retrieval_engine_ms",
            "groq_ms",
            "answer_translation_ms",
        ],
    },
    "query": {
        "path": "/v1/query",
        "total_key": "total_turn_ms",
        "stage_keys": [
            "guardrail_check_ms",
            "retrieval_embedding_ms",
            "retrieval_qdrant_ms",
            "retrieval_fusion_ms",
            "retrieval_total_ms",
            "groq_call_ms",
            "text_trim_ms",
        ],
    },
}


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class RequestResult:
    query: str
    language_code: str
    client_wall_ms: float
    http_status: int
    ok: bool
    error: Optional[str] = None
    server_timings: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# BENCHMARK RUNNER
# ============================================================

def build_payload(endpoint_name: str, text: str, language_code: str) -> Dict[str, str]:
    if endpoint_name == "text":
        return {"text": text, "language_code": language_code}
    # /v1/query is English-only per project/api/app.py (QueryRequest.query)
    return {"query": text}


def run_one(session: requests.Session, base_url: str, endpoint_name: str,
            text: str, language_code: str, timeout_s: float) -> RequestResult:
    cfg = ENDPOINTS[endpoint_name]
    url = base_url.rstrip("/") + cfg["path"]
    payload = build_payload(endpoint_name, text, language_code)

    start = time.perf_counter()
    try:
        resp = session.post(url, json=payload, timeout=timeout_s)
        client_wall_ms = (time.perf_counter() - start) * 1000.0

        if resp.status_code != 200:
            return RequestResult(
                query=text, language_code=language_code,
                client_wall_ms=client_wall_ms, http_status=resp.status_code,
                ok=False, error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        body = resp.json()
        timings = body.get("timings", {}) or {}

        return RequestResult(
            query=text, language_code=language_code,
            client_wall_ms=client_wall_ms, http_status=resp.status_code,
            ok=True, server_timings=timings,
        )

    except requests.RequestException as exc:
        client_wall_ms = (time.perf_counter() - start) * 1000.0
        return RequestResult(
            query=text, language_code=language_code,
            client_wall_ms=client_wall_ms, http_status=-1,
            ok=False, error=str(exc),
        )


def build_test_plan(repeats: int, include_multilingual: bool,
                     endpoint_name: str, seed: int) -> List[Dict[str, str]]:
    """Builds the ordered list of (text, language_code) requests to fire.
    Each question is repeated `repeats` times so percentile stats aren't
    driven by any single lucky/unlucky run, then the whole plan is
    shuffled so repeated hits on the same question aren't back-to-back
    (which would bias results via caching/keep-alive warmth)."""
    plan: List[Dict[str, str]] = []

    for q in QUESTIONS:
        for _ in range(repeats):
            plan.append({"text": q, "language_code": "en-IN"})

    if include_multilingual and endpoint_name == "text":
        for item in MULTILINGUAL_QUERIES:
            for _ in range(repeats):
                plan.append(dict(item))

    rng = random.Random(seed)
    rng.shuffle(plan)
    return plan


# ============================================================
# STATISTICS
# ============================================================

def percentile(values: List[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy.percentile default),
    implemented without a numpy dependency."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p50": percentile(values, 50),
        "p70": percentile(values, 70),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p100": max(values),  # p100 == max, called out explicitly per spec
    }


# ============================================================
# REPORTING
# ============================================================

def print_summary_table(title: str, stats: Dict[str, float]) -> None:
    if not stats:
        print(f"{title}: no data")
        return
    print(f"\n{title}  (n={stats['n']})")
    print("-" * 72)
    print(
        f"  mean={stats['mean']:8.1f}ms  stdev={stats['stdev']:7.1f}ms  "
        f"min={stats['min']:8.1f}ms"
    )
    print(
        f"  P50 ={stats['p50']:8.1f}ms  "
        f"P70 ={stats['p70']:8.1f}ms  "
        f"P90 ={stats['p90']:8.1f}ms  "
        f"P95 ={stats['p95']:8.1f}ms  "
        f"P100={stats['p100']:8.1f}ms"
    )


def write_raw_csv(path: Path, results: List[RequestResult], stage_keys: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["query", "language_code", "ok", "http_status", "client_wall_ms", "error"]
            + stage_keys
        )
        for r in results:
            writer.writerow(
                [r.query, r.language_code, r.ok, r.http_status,
                 f"{r.client_wall_ms:.2f}", r.error or ""]
                + [r.server_timings.get(k, "") for k in stage_keys]
            )


def write_summary_json(path: Path, endpoint_name: str,
                        client_stats: Dict[str, float],
                        server_total_stats: Dict[str, float],
                        stage_stats: Dict[str, Dict[str, float]],
                        error_count: int, total_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "endpoint": endpoint_name,
        "total_requests": total_count,
        "errors": error_count,
        "error_rate_pct": round(100.0 * error_count / total_count, 2) if total_count else 0.0,
        "client_wall_ms": client_stats,
        "server_total_ms": server_total_stats,
        "stage_ms": stage_stats,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def maybe_plot(path: Path, client_values: List[float], stats: Dict[str, float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping chart; `pip install matplotlib` to enable)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.hist(client_values, bins=30, color="#4C72B0", edgecolor="white")
    ax1.set_title("End-to-end latency distribution")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Requests")

    labels = ["P50", "P70", "P90", "P95", "P100"]
    vals = [stats["p50"], stats["p70"], stats["p90"], stats["p95"], stats["p100"]]
    bars = ax2.bar(labels, vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"])
    ax2.set_title("Latency percentiles")
    ax2.set_ylabel("Latency (ms)")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"\nSaved chart -> {path}")


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Latency benchmark for the HHGOa pipeline")
    parser.add_argument("--base-url", default="http://localhost:8000",
                         help="Backend base URL (default: http://localhost:8000)")
    parser.add_argument("--endpoint", choices=list(ENDPOINTS.keys()), default="text",
                         help="Which pipeline endpoint to benchmark (default: text)")
    parser.add_argument("--repeats", type=int, default=5,
                         help="How many times each question is sent (default: 5 -> 50 requests over the 10-question set)")
    parser.add_argument("--warmup", type=int, default=5,
                         help="Number of untimed warm-up requests before the timed run (default: 5)")
    parser.add_argument("--timeout", type=float, default=30.0,
                         help="Per-request timeout in seconds (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Shuffle seed, for reproducible test order (default: 42)")
    parser.add_argument("--no-multilingual", action="store_true",
                         help="Exclude the Hindi sample queries (only affects --endpoint text)")
    parser.add_argument("--out", default="results/latency_run",
                         help="Output path prefix for raw CSV / summary JSON / chart")
    parser.add_argument("--no-plot", action="store_true", help="Skip chart generation")
    args = parser.parse_args()

    cfg = ENDPOINTS[args.endpoint]

    plan = build_test_plan(
        repeats=args.repeats,
        include_multilingual=not args.no_multilingual,
        endpoint_name=args.endpoint,
        seed=args.seed,
    )

    print("=" * 90)
    print("PIPELINE LATENCY BENCHMARK")
    print("=" * 90)
    print(f"Endpoint        : POST {args.base_url}{cfg['path']}")
    print(f"Distinct queries: {len(QUESTIONS) + (0 if args.no_multilingual or args.endpoint != 'text' else len(MULTILINGUAL_QUERIES))}")
    print(f"Repeats/query   : {args.repeats}")
    print(f"Total requests  : {len(plan)}")
    print(f"Warm-up requests: {args.warmup} (untimed, excluded from stats)")
    print()

    session = requests.Session()

    # ---- Warm-up (model loading, connection pooling, JIT caches, etc.) ----
    if args.warmup > 0:
        print("Warming up...")
        for i in range(args.warmup):
            item = plan[i % len(plan)]
            run_one(session, args.base_url, args.endpoint,
                    item["text"], item["language_code"], args.timeout)
        print("Warm-up done.\n")

    # ---- Timed run ----
    results: List[RequestResult] = []
    for i, item in enumerate(plan, start=1):
        r = run_one(session, args.base_url, args.endpoint,
                     item["text"], item["language_code"], args.timeout)
        results.append(r)
        status = "OK  " if r.ok else "FAIL"
        print(f"[{i:03d}/{len(plan)}] {status}  {r.client_wall_ms:8.1f}ms  | {item['text'][:60]}")

    # ---- Aggregate ----
    ok_results = [r for r in results if r.ok]
    error_results = [r for r in results if not r.ok]

    client_values = [r.client_wall_ms for r in ok_results]
    server_total_values = [
        r.server_timings[cfg["total_key"]]
        for r in ok_results
        if cfg["total_key"] in r.server_timings and r.server_timings[cfg["total_key"]] is not None
    ]

    client_stats = summarize(client_values)
    server_total_stats = summarize(server_total_values)

    stage_stats: Dict[str, Dict[str, float]] = {}
    for key in cfg["stage_keys"]:
        vals = [
            r.server_timings[key]
            for r in ok_results
            if r.server_timings.get(key) not in (None, "")
        ]
        if vals:
            stage_stats[key] = summarize(vals)

    # ---- Report ----
    print("\n")
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)
    print(f"Successful requests : {len(ok_results)}/{len(results)}")
    print(f"Failed requests     : {len(error_results)}")
    if error_results:
        print("Sample errors:")
        for r in error_results[:5]:
            print(f"  - [{r.http_status}] {r.query[:50]!r}: {r.error}")

    print_summary_table("CLIENT-MEASURED END-TO-END LATENCY (submit these numbers)", client_stats)
    print_summary_table(f"SERVER-REPORTED TOTAL ({cfg['total_key']})", server_total_stats)

    print("\nPer-stage breakdown (server-reported, ms):")
    print("-" * 72)
    for key, stats in stage_stats.items():
        print(f"  {key:<28} P50={stats['p50']:7.1f}  P70={stats['p70']:7.1f}  "
              f"P90={stats['p90']:7.1f}  P100={stats['p100']:7.1f}  (n={stats['n']})")

    # ---- Persist ----
    out_prefix = Path(args.out)
    write_raw_csv(out_prefix.with_suffix(".raw.csv"), results, cfg["stage_keys"])
    write_summary_json(
        out_prefix.with_suffix(".summary.json"),
        args.endpoint, client_stats, server_total_stats, stage_stats,
        error_count=len(error_results), total_count=len(results),
    )
    print(f"\nSaved raw results  -> {out_prefix.with_suffix('.raw.csv')}")
    print(f"Saved summary JSON -> {out_prefix.with_suffix('.summary.json')}")

    if not args.no_plot and client_values:
        maybe_plot(out_prefix.with_suffix(".png"), client_values, client_stats)

    return 0 if not error_results else 1


if __name__ == "__main__":
    sys.exit(main())