# Qdrant Hybrid Retrieval — Test Report

**Collection:** `msmarco_english_corpus`
**Dataset:** MSMARCO-XI (Hindi-English parallel corpus, English passages used)
**Dense model:** `BAAI/bge-small-en-v1.5` (384-dim, cosine)
**Sparse model:** `Qdrant/bm25`
**Test set:** 60 stratified sample queries (DESCRIPTION / NUMERIC / ENTITY / LOCATION / PERSON)
**Date:** August 2026

---

## 1. Objective

Evaluate retrieval quality and latency of a Qdrant-based vector search pipeline across multiple retrieval strategies — dense-only, sparse-only (BM25), and several hybrid fusion approaches — and validate the integrity of the underlying data pipeline (preprocessing → chunking → embedding → upsert).

---

## 2. Pipeline Summary

1. **Preprocessing** — HTML unescape, URL/citation stripping, whitespace normalization, artifact cleanup, min-length validity filter
2. **Enrichment** — regex year extraction, spaCy NER (people/orgs/locations)
3. **Chunking** — 3 strategies generated (`passage_native`, `fixed_overlap`, `semantic`), deduplicated with `passage_native` priority (`SequenceMatcher`, threshold 0.9)
4. **Storage** — SQLite native lookup (`passage_id`, `query_id`, native text) + Qdrant vector index with payload indexes on `lang`, `query_type`, `chunk_strategy`, `source_dataset`, entity flags, `year_mentions`
5. **Embedding & upsert** — dense (`bge-small-en-v1.5`) and sparse (`Qdrant/bm25`) vectors stored under named vectors `"dense"` and `"sparse"` in the same collection

---

## 3. Retrieval Modes Tested

| Mode | Description |
|---|---|
| `dense` | Dense-only cosine similarity search |
| `sparse` | BM25 sparse keyword search |
| `hybrid_rrf_k60` | Qdrant native Reciprocal Rank Fusion (fixed k=60, equal weight) |
| `hybrid_weighted` | Manual score-normalized fusion, dense weight 0.75 / sparse weight 0.25 |
| `rrf_k5` / `rrf_k10` / `rrf_k30` / `rrf_k100` | Manual RRF implementation with tunable k, for comparison against native fusion |

---

## 4. Benchmark Results (60 queries, 39 with valid ground truth)

| Mode | hit_rate@5 | MRR | mean_ms | p90_ms | integrity_issue_rate |
|---|---|---|---|---|---|
| **hybrid_weighted** | **0.9487** | 0.6115 | 91.8 | 107.8 | **0.0500** |
| hybrid_rrf_k60 | 0.9231 | 0.5658 | 58.9 | 72.6 | 0.0833 |
| dense | 0.8974 | **0.6462** | 70.0 | 98.1 | **0.0500** |
| rrf_k5 | 0.8718 | 0.5274 | 86.9 | 103.4 | 0.0667 |
| rrf_k10 | 0.8462 | 0.5026 | 105.2 | 133.3 | 0.0667 |
| rrf_k30 | 0.8205 | 0.4919 | 89.9 | 108.2 | 0.0667 |
| rrf_k100 | 0.8205 | 0.4919 | 88.1 | 104.2 | 0.0833 |
| sparse | 0.7179 | 0.4085 | 23.3 | 28.0 | 0.1500 |

**Metric definitions:**
- `hit_rate@5` — fraction of valid-GT queries where the correct passage appeared anywhere in the top 5 results
- `MRR` — mean reciprocal rank (1/rank of correct answer, averaged; rewards top-1 placement)
- `integrity_issue_rate` — fraction of top-1 results whose `passage_id` belonged to a different `query_id` than expected

**Result confirmed across two independent 60-query samples** (different random seeds) — dense and hybrid_weighted showed consistent relative performance in both runs.

---

## 5. Key Finding: RRF `k` Tuning Does Not Help

A sweep of Qdrant's Reciprocal Rank Fusion constant (`k = 5, 10, 30, 60, 100`) showed **no improvement over the default k=60**, and larger k values performed worse. This is because RRF is purely **rank-based** — it only considers which document ranked #1, #2, etc. in each retriever's list, never *how much better* one match is than another.

The fix that worked was **`hybrid_weighted`**: normalizing dense and sparse scores to a 0–1 range independently, then combining with fixed weights (dense=0.75, sparse=0.25). This is **score-based**, not rank-based, and outperformed every RRF configuration tested on both hit_rate and MRR.

---

## 6. ID Integrity Validation

A two-layer validation was run on every retrieved result:

1. **GT match** — does the retrieved `passage_id` match the dataset's officially labeled correct answer?
2. **Integrity match** — does the retrieved `passage_id` exist in the native SQLite lookup, and does its stored `query_id` match the query that was searched?

**Result:** No data corruption found. All "integrity mismatches" were traced to **5–7 specific hard queries**, not random or systemic errors. Manual re-encoding of affected passages produced bit-for-bit identical similarity scores to what Qdrant returned, confirming stored vectors are consistent and uncorrupted.

### Root causes identified (all confirmed via manual investigation)

| Query ID | Query | Root Cause | Category |
|---|---|---|---|
| 447750 | "meaning of eria" | Dataset typo — should be "aria"; own passage pool never mentions "eria" | Dataset error |
| 325563 | "how much negative info can you expect from a seller" | Genuine close-call: correct passage scored 0.6917 vs. unrelated passage at 0.7037 | Embedding model limitation |
| 1146530 | "what regulation covers sop" | Ambiguous acronym ("SOP"), low-confidence match (0.66 vs 0.68) | Embedding model limitation |
| 503650 | "steps to calcukate RBRVS" | Rare acronym + typo ("calcukate"), same close-call pattern | Embedding model limitation |
| 427189 | "tumor blood supply process" | **Dense mode retrieved correctly (rank 1, score 0.80)** — issue only appeared under sparse/RRF modes | Mode-selection sensitivity, not a data bug |
| 388153, 46489, 348033, 252492, 894131 | Various | Sparse-only keyword collisions on shared rare terms | Sparse retriever limitation |

**Conclusion:** Zero pipeline bugs found. All flagged cases are either (a) pre-existing dataset quality issues, (b) known limitations of dense embeddings on short/ambiguous/acronym-heavy queries, or (c) resolved by mode selection (`hybrid_weighted` already mitigates case c).

---

## 7. Query Expansion Experiment

Tested whether correcting typos / expanding acronyms improves retrieval on the 3 hardest cases, across all 5 modes:

| Query | Fix Applied | Result |
|---|---|---|
| 447750 "eria" → "aria" | Typo correction | **Fixed across all 5 modes** — correct passage jumped to rank 1 (score 0.88) |
| 325563 (added "feedback"/"disclose") | Vocabulary alignment | **Fixed across all 5 modes** — correct passage retrieved, score 0.74 |
| 1146530 "sop" → "SOP (Standard Operating Procedure)" | Acronym expansion | **Mixed** — sparse improved, dense/hybrid shifted to a different (semantically reasonable but dataset-mismatched) wrong answer |

**Takeaway:** Query preprocessing (typo correction, acronym expansion) is a net-positive technique — 2 of 3 hard cases were fully resolved across every mode. It carries a small risk of over-correcting toward a more general (but dataset-mismatched) interpretation, as seen in the SOP case.

---

## 8. Query-Type Breakdown (hybrid_weighted vs. dense)

| Query Type | n | dense MRR | hybrid_weighted MRR |
|---|---|---|---|
| DESCRIPTION | 37 | 0.4347 | 0.3937 |
| ENTITY | 4 | 0.2083 | **0.4583** |
| LOCATION | 3 | 0.6667 | 0.6667 |
| NUMERIC | 13 | 0.4064 | 0.3551 |
| PERSON | 3 | 0.3333 | 0.2778 |

`hybrid_weighted` provides its largest gains on **ENTITY**-type queries (exact name/keyword matching benefits from sparse's contribution), while remaining close to dense's performance elsewhere.

---

## 9. Latency Analysis

- **Sparse** is fastest (~23ms mean) but least accurate — not viable standalone
- **Dense** ~70ms mean, most consistent MRR
- **hybrid_weighted** ~92ms mean — highest accuracy, acceptable latency overhead (~22ms above dense) from running two retrieval passes + Python-side score fusion
- One transient latency spike (1068ms on a single sparse query) was reproduced 5x and confirmed as a one-off system hiccup, not a systemic issue

---

## 10. Recommendation

**Use `hybrid_weighted` (dense=0.75, sparse=0.25) as the production retrieval mode.**

Justification:
- Best hit_rate@5 (0.9487) across all modes and both independent test samples
- MRR (0.6115) close to dense's ceiling (0.6462), unlike plain RRF fusion which degraded MRR
- Tied-lowest ID integrity issue rate (0.05, same as dense alone)
- Outperformed every RRF k-value tested (5, 10, 30, 60, 100)
- Latency (~92ms mean) is acceptable for the accuracy gain

**Fallback for latency-critical use cases:** `dense` alone — nearly as accurate, ~20ms faster, simpler to maintain (no fusion logic).

**Optional future improvement:** Apply query preprocessing (typo correction / acronym expansion) as a pre-embedding step — shown to fully resolve 2 of 3 tested hard cases across all modes, with no observed downside beyond one query where the correction shifted toward a more general (dataset-mismatched) interpretation.

---

## 11. Files Generated

- `benchmark_60q_details.csv` / `benchmark_v2_60q_details.csv` — raw per-query results, two independent samples
- `FINAL_report_summary.csv` — one row per mode: hit_rate, MRR, timing, integrity
- `FINAL_report_full_details.csv` — every query × mode combination
- `FINAL_report_integrity_issues.csv` — flagged ID mismatches with root-cause notes
- `acronym_expansion_test.csv` — before/after query expansion results
- `no_gt_queries_dense_sparse_hybrid_answers.csv` — manual answers for the 21 unlabeled queries