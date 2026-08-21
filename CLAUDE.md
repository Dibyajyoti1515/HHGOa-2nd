# CLAUDE.md

> Context/handoff doc for this project. If you are a different Claude instance/session picking this up, read this fully before touching code.

## 0. Source of truth

**Source notebook:** `Update_version_2.ipynb` (uploaded by the user).

**Hard rule for anyone working on this repo:** No new retrieval, chunking, embedding, dedup, or scoring logic may be invented. Every function in this codebase must trace back to a specific cell in that notebook. The notebook is messy (iterative, has dead ends, has a dense-only version later superseded by a hybrid version, has ~15 benchmark/diagnostic cells at the end) — the job here is **modularization and productionization**, not redesign.

Only the following are genuinely new (not in the notebook) and were built fresh for this project:
- Guardrails (input validation layer)
- Logging system
- Timing/instrumentation system
- FastAPI serving layer / orchestration
- LLM fallback wiring (Groq) and TTS wiring (ElevenLabs) — control-flow only, per user's own draft `app.py`/`vector_ingest.py`

## 1. What the notebook actually contains

Two logically separate jobs live in one notebook:

### Job A — Ingestion (build the index)
Cells 1–18, run once, in order:
1. Config constants (sample size, query type distribution, model names, thresholds)
2. Parquet download/stream-read helpers
3. Stratified sampling by `query_type`
4. Explode MSMARCO-XI rows into per-passage rows (native + translated text kept)
5. Preprocessing (HTML unescape, URL strip, whitespace normalize, min-length filter)
6. Enrichment (regex year extraction + spaCy NER for PERSON/ORG/GPE-LOC)
7. Three chunking strategies: `passage_native`, `fixed_overlap` (256 tok / 20% overlap), `semantic` (sentence-similarity grouping, 128–320 tok bounds)
8. Dedup across all three strategies per `passage_id`, priority: native > semantic > fixed_overlap (`SequenceMatcher` ratio > 0.9 = duplicate)
9. Write `unified_corpus.parquet` (full archive) + `native_text_lookup.db` (SQLite, English-language passages only, indexed by `passage_id`/`query_id`)
10. Embed + upsert to Qdrant — **this step was iterated on twice:**
    - First version: dense-only (`bge-small-en-v1.5`, 384-dim, cosine), single `vectors_config`
    - **Superseded version:** hybrid — same dense model + sparse `Qdrant/bm25` (via `fastembed`), collection recreated with named vectors `{"dense": ..., "sparse": ...}`, 9 payload indexes (lang, query_type, chunk_strategy, source_dataset, has_year/person/org/location, year_mentions)

**Only the hybrid ingestion path is being ported.** The dense-only `embed_and_upsert` function is superseded and dropped.

### Job B — Retrieval (query the index)
Scattered across cells post-Cell 18, cleaned up and finalized in **Cell 26-DEFS** and **Cell 39**:

| Function | Mode | Notebook cell |
|---|---|---|
| `retrieve_top_k_dense_only` | dense vector search only | Cell 26-DEFS |
| `retrieve_top_k_sparse_only` | BM25 sparse search only | Cell 26-DEFS |
| `retrieve_top_k_hybrid_only` | Qdrant-native RRF fusion (dense+sparse, prefetch limit 20 each) | Cell 26-DEFS |
| `retrieve_top_k_hybrid_weighted` | manual min-max normalize + weighted sum (dense 0.75 / sparse 0.25, candidate pool 20) | Cell 39 |
| `retrieve_top_k_rrf_tunable` | manual RRF with tunable `k` (5/10/30/60/100) | Cell 45 |

### The rest of the notebook (Cells 19–50, excluding 26-DEFS/39 above)
Benchmarking, ID-integrity checks, acronym-expansion experiments, a second 60-query test set, and a full 8-mode comparison (Cell 46/48). **This is not being ported into the codebase.** Per user decision, no `benchmarks/` folder is being created — this was exploratory work used to pick the production retrieval mode, and its job is already done (see Decision Log below). If someone needs to re-run this analysis later, go back to the notebook directly.

## 2. Decisions made (do not re-litigate without user sign-off)

| # | Decision | Rationale |
|---|---|---|
| 1 | Default retrieval mode = **`hybrid_weighted`** | Best hit_rate@5 (0.9487) across the notebook's own Cell 48 benchmark (39 queries with valid ground truth). Other 3 modes (`dense`, `hybrid_rrf`, `rrf_tunable`) are kept in the codebase and selectable via env var for comparison, but not exposed as user-facing options. |
| 2 | Ingestion scope = **English only** | User confirmed: no Odia/Telugu, matches what the notebook actually ran end-to-end (only `hi_norm`/Hindi-sourced-but-English-target rows were ever fully processed — the notebook's own multi-language hook (`pd.concat([hi_norm])`, "add Odia/Telugu here later if needed") was never used). |
| 3 | Confidence gate threshold = **0.85** on `top1_score` | `< 0.85` → Groq LLM composes an answer from retrieved context, then TTS. `>= 0.85` → skip LLM, just sentence-boundary-trim retrieved text to 200 words, then TTS directly. This matches the user's own draft `app.py`. |
| 4 | Guardrails run **before** retrieval, return **HTTP 404** on failure (temporary — see open item below) | Checks (this phase only): empty/too-long query, prompt-injection patterns, PII, profanity, blocked topics, plus a small set of obvious catch-alls (see Guardrails section). Audio-file guardrail checks are explicitly deferred — "after that we implement the audio file" — i.e. this phase is text-input guardrails only. |
| 5 | Timing report format = **JSON**, shape `{ audio_url, timings: {...} }` | Every stage of the request gets its own millisecond entry: guardrail check, query embedding (dense), query embedding (sparse), Qdrant search, [Groq call if triggered], text trim, TTS generation, total. |
| 6 | No `benchmarks/` folder in the repo | User's explicit instruction. Benchmark/diagnostic cells stay in the notebook only. |
| 7 | Guardrails config = **file-based wordlists**, not env-var lists | User supplied `guardrails/input_guardrails.py` directly (rule-based, stdlib-only: empty/too-long, non-UTF8/garbage, script/HTML injection, prompt-injection, PII, profanity, blocked topics — covers every item from Decision #4 plus the agreed catch-alls). It expects `GUARDRAIL_PROFANITY_FILE` / `GUARDRAIL_BLOCKED_TOPICS_FILE` (one term per line, `#`=comment, empty file = that check is a no-op) and `GUARDRAIL_RESPONSE_DETAIL` (`"detailed"` or `"generic"`) rather than the list/bool shape originally planned. `config/settings.py` was updated to match — this **supersedes** the original 404-body and blocked-topics-list open items below (both are now resolved: detail is `"detailed"` by default, revealing which check failed; blocked topics are an empty stub file at `guardrails/wordlists/blocked_topics.txt` for the user to fill in). Only the file's import block was changed; all check logic is exactly as supplied. |
| 8 | Timing granularity for retrieval = **not split into dense/sparse** | `retrieve_top_k_hybrid_weighted` (Cell 39) measures dense-encode + sparse-embed together as a single `embedding_ms` (one timer spans both calls in the notebook). Splitting that would mean editing the ported retrieval function, which Decision/Rule #0 forbids. `api/app.py`'s timings block reports `retrieval_embedding_ms` (combined), `retrieval_qdrant_ms`, `retrieval_fusion_ms`, `retrieval_total_ms` instead of the originally sketched `dense_embedding_ms`/`sparse_embedding_ms` split in Section 7. |

## 3. Open items — RESOLVED

~~Guardrails detail spec~~ — resolved by Decision #7 above (user supplied the module directly).
~~Blocked topics list~~ — resolved: stub file at `guardrails/wordlists/blocked_topics.txt`, empty by default, user fills in.
~~404 response body shape~~ — resolved: `GUARDRAIL_RESPONSE_DETAIL=detailed` by default (names the failed check), can be set to `"generic"` for a flat error.
~~Detection method~~ — resolved: rule/regex-based, stdlib only, as originally defaulted to.

No open items remain from the original build plan. The only explicitly deferred item is **audio-output guardrails** (Decision #4), which was never in scope for this phase.

## 4. File system layout

```
project/
├── CLAUDE.md
├── .env.example
├── .gitignore
├── requirements.txt
│
├── config/
│   └── settings.py                  # Cell 1 constants + env loading (model names, thresholds, paths, collection name)
│
├── ingestion/
│   ├── download.py                  # download_language_file, stream_read_parquet        [Cell 2]
│   ├── sampling.py                  # stratified_sample                                   [Cell 3]
│   ├── explode.py                   # explode_msmarco_xi                                  [Cell 4]
│   ├── preprocess.py                # preprocess_text, is_valid_row, preprocess_corpus    [Cell 5]
│   ├── enrich.py                    # enrich (spaCy NER + YEAR_RE)                        [Cell 6]
│   ├── chunking/
│   │   ├── native.py                # chunk_passage_native                                [Cell 7]
│   │   ├── fixed_overlap.py         # chunk_fixed_overlap                                 [Cell 7]
│   │   ├── semantic.py              # chunk_semantic                                      [Cell 7]
│   │   └── tagging.py               # tag_metadata_aware                                  [Cell 7]
│   ├── dedup.py                     # dedupe_chunks, STRATEGY_PRIORITY                    [Cell 8]
│   ├── storage/
│   │   ├── sqlite_writer.py         # build_native_lookup                                 [Cell 9]
│   │   └── parquet_writer.py        # write_unified_file                                  [Cell 9]
│   ├── embeddings/
│   │   ├── dense_model.py           # dense model loader (SentenceTransformer)            [Cell 10 / hybrid version]
│   │   └── sparse_model.py          # sparse model loader (fastembed SparseTextEmbedding, Qdrant/bm25)
│   ├── qdrant_setup.py              # collection create/recreate + payload index creation [hybrid version]
│   ├── upsert.py                    # embed_and_upsert_hybrid split into: embed_dense, embed_sparse, build_points, upsert_batch
│   └── pipeline.py                  # orchestrator — chains all of the above, mirrors Cells 11→18 execution order
│
├── retrieval/
│   ├── dense_retriever.py           # retrieve_top_k_dense_only            [Cell 26-DEFS]
│   ├── sparse_retriever.py          # retrieve_top_k_sparse_only           [Cell 26-DEFS]
│   ├── hybrid_rrf_retriever.py      # retrieve_top_k_hybrid_only           [Cell 26-DEFS]
│   ├── hybrid_weighted_retriever.py # retrieve_top_k_hybrid_weighted       [Cell 39]  ← PRODUCTION DEFAULT
│   └── rrf_tunable_retriever.py     # retrieve_top_k_rrf_tunable           [Cell 45]  (kept, not default)
│
├── guardrails/
│   ├── input_guardrails.py          # NEW — user-supplied module (see Decision #7): empty/too-long,
│   │                                 #       non-UTF8/garbage, script/HTML injection, prompt-injection,
│   │                                 #       PII, profanity, blocked topics. Only its import block was edited.
│   └── wordlists/
│       ├── profanity.txt            # one term per line, '#'=comment, empty=no-op (stub, user fills in)
│       └── blocked_topics.txt       # same format (stub, user fills in)
│
├── llm_fallback/
│   └── groq_client.py               # call_groq — from user's draft app.py
│
├── tts/
│   ├── text_trim.py                 # trim_to_word_limit (sentence-boundary trim) — from user's draft app.py
│   └── elevenlabs_client.py         # call_elevenlabs_tts — from user's draft app.py
│
├── logging_system/
│   └── logger.py                    # NEW — central logger config (file + console handlers, structured format)
│
├── timing/
│   └── stage_timer.py               # NEW — reusable timer (context manager/decorator) feeding the timings JSON block
│
├── api/
│   └── app.py                       # FastAPI: POST /query
│                                     #   flow: guardrails → embed query → hybrid_weighted retrieval →
│                                     #         confidence gate (0.85) → [groq | trim] → tts →
│                                     #         response { audio_url, timings }
│
├── scripts/
│   └── run_ingestion.py             # CLI entrypoint: python -m scripts.run_ingestion --input <parquet> --text-col text_en
│
└── tests/
```

Note: no `benchmarks/` directory — see Decision #6.

## 5. Guardrails — implemented (`guardrails/input_guardrails.py`)

Runs on the raw text query, **before** anything touches retrieval/embedding. Implemented as `run_guardrails_strict(query)`, which raises `GuardrailViolation` on the first failed check (checks run in order, short-circuit on first failure). `api/app.py` catches that and returns **HTTP 404** with `build_error_response(exc)` as the body.

Checks, in execution order:
1. `check_empty` — empty/whitespace-only query
2. `check_too_long` — over `settings.GUARDRAIL_MAX_QUERY_LENGTH` (default 500 chars)
3. `check_non_utf8_garbage` — invalid UTF-8, or >10% control characters
4. `check_script_html_injection` — `<script>`, `<iframe>`, `javascript:`, inline event handlers, etc.
5. `check_prompt_injection` — rule-based pattern list ("ignore previous instructions", "you are now a...", "jailbreak", "developer mode", etc.)
6. `check_pii` — email, phone, SSN, credit-card-shaped patterns
7. `check_profanity` — wordlist file, empty by default (no-op)
8. `check_blocked_topics` — wordlist file, empty by default (no-op)

404 body shape is controlled by `settings.GUARDRAIL_RESPONSE_DETAIL`:
- `"detailed"` (default): `{"error": "guardrail_violation", "check": "<name>", "detail": "<message>"}`
- `"generic"`: `{"error": "Request blocked by input validation."}`

Explicitly **out of scope for this phase**: any guardrail on the generated audio output. That comes later, once the audio pipeline itself is wired up, per user instruction.

## 6. Logging system — requirements

- Centralized logger config in `logging_system/logger.py`, imported everywhere else (no ad-hoc `print()` in production code).
- Must log: ingestion pipeline progress (per stage, per batch), every retrieval request (query, mode used, top1 score, hit stage), every guardrail rejection (which check failed), every Groq fallback trigger, every TTS generation, all errors/exceptions with stack traces.
- File-based logging (rotating or per-run — to be decided when this module is built) plus console output.

## 7. Timing system — implemented (`timing/stage_timer.py`, wired in `api/app.py`)

Every request returns, alongside `audio_url`:

```json
{
  "audio_url": "/audio/<uuid>.mp3",
  "timings": {
    "guardrail_check_ms": ...,
    "retrieval_embedding_ms": ...,   // dense+sparse encode, combined (see Decision #8 — not split)
    "retrieval_qdrant_ms": ...,
    "retrieval_fusion_ms": ...,
    "retrieval_total_ms": ...,
    "groq_call_ms": null,            // null when confidence gate skipped the LLM; a float otherwise
    "text_trim_ms": null,            // null when Groq fallback was used; a float otherwise
    "tts_generation_ms": ...,
    "total_ms": ...                  // true request wall-clock, not a sum of the parts above
  }
}
```

`TimingCollector` (in `timing/stage_timer.py`) reuses the `*_ms` values that retrieval/Groq/trim/TTS functions already compute and return internally — it doesn't re-measure anything, it just collects. The one stage that isn't self-timed elsewhere (guardrails) is measured via `TimingCollector.stage(name)`, a context manager built for exactly that purpose.

## 8. Environment variables (`.env`)

To be finalized in `config/settings.py`, but known from the notebook + draft files so far:

```
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=msmarco_english_corpus
EMBED_MODEL=BAAI/bge-small-en-v1.5
EMBED_DIM=384
SPARSE_MODEL=Qdrant/bm25
DEDUP_THRESHOLD=0.9
SAMPLE_SIZE_PER_LANG=3000

RETRIEVAL_MODE=hybrid_weighted
CONFIDENCE_THRESHOLD=0.85
HYBRID_DENSE_WEIGHT=0.75
HYBRID_SPARSE_WEIGHT=0.25
HYBRID_CANDIDATE_POOL=20

GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile   # verify current model name in Groq console — may be deprecated

ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=

TTS_WORD_LIMIT=200

LOG_LEVEL=INFO
LOG_DIR=./logs

GUARDRAIL_MAX_QUERY_LENGTH=500
GUARDRAIL_MIN_QUERY_LENGTH=1
GUARDRAIL_PROFANITY_FILE=guardrails/wordlists/profanity.txt
GUARDRAIL_BLOCKED_TOPICS_FILE=guardrails/wordlists/blocked_topics.txt
GUARDRAIL_RESPONSE_DETAIL=detailed
```

See `.env.example` in the repo root for the authoritative, always-current copy of this list.

## 9. Build order (phases) — ALL COMPLETE

1. ✅ `CLAUDE.md` (this file)
2. ✅ `config/settings.py` + `.env.example`
3. ✅ `ingestion/` modules (one file per notebook function) → `ingestion/pipeline.py` orchestrator
4. ✅ `retrieval/` modules (all 5 functions — dense, sparse, hybrid_rrf, hybrid_weighted, rrf_tunable — `hybrid_weighted` as default) + `retrieval/factory.py` dispatcher
5. ✅ `guardrails/input_guardrails.py` — user supplied this module directly; only its import block was changed (see Decision #7 below)
6. ✅ `llm_fallback/groq_client.py` + `tts/text_trim.py` + `tts/elevenlabs_client.py`
7. ✅ `logging_system/logger.py` (built early, ahead of schedule, since ingestion modules needed it to replace notebook `print()` calls)
8. ✅ `timing/stage_timer.py`
9. ✅ `api/app.py` — wires everything together
10. ✅ `scripts/run_ingestion.py` — CLI entrypoint
11. ✅ `requirements.txt`

Remaining open item: **audio-output guardrails** — explicitly deferred, not part of this build (see Decision #4 and Section 5).

## 10. What NOT to do

- Do not invent new chunking/scoring/fusion logic not present in the notebook.
- Do not add Odia/Telugu or any multi-language ingestion hooks — English only, per user decision.
- Do not build audio-output guardrails yet.
- Do not create a `benchmarks/` folder or port Cells 19–50 wholesale.
- Do not silently pick a blocked-topics list or a 404 error body shape — these are open items requiring user sign-off.