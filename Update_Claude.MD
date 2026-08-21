# CLAUDE.md

> Context/handoff doc for this project. If you are a different Claude instance/session picking this up, read this fully before touching code.

## 0. Source of truth

**Source notebook:** `Update_version_2.ipynb` (uploaded by the user).

**Hard rule for anyone working on this repo:** No new retrieval, chunking, embedding, dedup, or scoring logic may be invented. Every function in that part of the codebase must trace back to a specific cell in that notebook. The notebook is messy (iterative, has dead ends, has a dense-only version later superseded by a hybrid version, has ~15 benchmark/diagnostic cells at the end) — the job there is **modularization and productionization**, not redesign.

This rule applies to `ingestion/` and `retrieval/` only. It does **not** apply to voice I/O (STT/TTS), guardrails, logging, timing, or the API/orchestration layer — those are genuinely new, hand-built components, same category as the guardrails/logging/timing systems below.

Only the following are genuinely new (not in the notebook) and were built fresh for this project:
- Guardrails (input validation layer)
- Logging system
- Timing/instrumentation system
- FastAPI serving layer / orchestration
- LLM fallback wiring (Groq) — control-flow only, per user's own draft `app.py`
- **STT — Sarvam AI realtime (websocket), with language detection** *(NEW — this phase)*
- **TTS — Sarvam AI realtime (websocket)** *(NEW — this phase, replaces ElevenLabs)*
- **React voice UI (record/pause/listen button)** *(NEW — this phase)*

## 0.1 ElevenLabs → Sarvam AI TTS migration (this phase)

**Decision:** TTS provider is being switched from ElevenLabs to **Sarvam AI realtime TTS**. `tts/elevenlabs_client.py` is being retired (kept in repo, unused, for reference only — not called from `api/`). `ELEVENLABS_API_KEY`/`ELEVENLABS_VOICE_ID` remain in `.env.example` but are no longer read by the live path.

> Assumption flagged for user sign-off: this reading treats "transfer it to sarvam AI" as a full swap, not a dual-provider setup. If ElevenLabs should stay as a fallback, say so and it'll be added as a secondary TTS path behind the same interface.

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
Benchmarking, ID-integrity checks, acronym-expansion experiments, a second 60-query test set, and a full 8-mode comparison (Cell 46/48). **This is not being ported into the codebase.** Per user decision, no `benchmarks/` folder is being created.

## 2. Decisions made (do not re-litigate without user sign-off)

| # | Decision | Rationale |
|---|---|---|
| 1 | Default retrieval mode = **`hybrid_weighted`** | Best hit_rate@5 (0.9487) across the notebook's own Cell 48 benchmark. |
| 2 | Ingestion scope = **English only** | User confirmed; matches what the notebook actually ran end-to-end. |
| 3 | Confidence gate threshold = **0.85** on `top1_score` | `< 0.85` → Groq composes answer, then TTS. `>= 0.85` → skip LLM, sentence-boundary-trim to 200 words, then TTS. |
| 4 | Guardrails run **before** retrieval, return **HTTP 404** on failure | Text-input guardrails only this phase; audio-output guardrails deferred. |
| 5 | Timing report format = **JSON**, shape `{ audio_url, timings: {...} }` | Extended this phase — see §7. |
| 6 | No `benchmarks/` folder in the repo | User's explicit instruction. |
| 7 | Guardrails config = **file-based wordlists**, not env-var lists | User-supplied `guardrails/input_guardrails.py`. |
| 8 | Timing granularity for retrieval = **not split into dense/sparse** | Single `retrieval_embedding_ms`. |
| 9 | **STT provider = Sarvam AI realtime API, over websocket** | User instruction. Frontend ↔ backend ↔ Sarvam AI is a **3-leg relay**: browser mic → backend websocket → Sarvam websocket. Backend does not do STT itself; it relays audio frames to Sarvam and relays partial/final transcripts back. Language detection is handled by Sarvam's `language-code: unknown` auto-detect mode — no separate language-ID step is built. |
| 10 | **TTS provider = Sarvam AI realtime API, over websocket** | Same 3-leg relay pattern, reversed: backend gets final answer text → opens websocket to Sarvam TTS → streams audio chunks back to frontend as they arrive (not batched into a single file), so playback can start before generation finishes. |
| 11 | **Voice pipeline is now the primary entrypoint**, `POST /query` (text-in) is kept as-is for non-voice/testing use | New endpoint `WS /voice` owns the full loop: STT → guardrails → retrieval → confidence gate → [Groq \| trim] → TTS. `POST /query` stays a thin text-only path for debugging without a mic. |
| 12 | **End-to-end timing now spans STT start → TTS last-chunk-sent**, not just the retrieval/LLM/TTS stages | See §7 for the extended timings shape. STT and TTS are both realtime/streaming, so each is timed as *time-to-first-partial*, *time-to-final-transcript*, and *time-to-first-audio-chunk* / *time-to-last-audio-chunk*, in addition to a single wall-clock duration — a single "STT took Xms" number would hide where the latency actually sits. |

## 3. Open items — mostly resolved, new ones flagged for this phase

Resolved (prior phase): guardrails detail spec, blocked topics list, 404 response body shape, detection method — see prior decisions above.

**New, needs user sign-off (proceeding with the stated default for now):**
- **Sarvam AI auth/model params** — exact websocket URL, auth header name, STT model name, TTS speaker/voice name, and sample rate are placeholders in `.env.example` (`SARVAM_STT_WS_URL`, `SARVAM_TTS_WS_URL`, `SARVAM_API_KEY`) — fill in from the Sarvam AI dashboard/docs before this runs against the real API. Client code is written against Sarvam's documented realtime STT/TTS websocket message shapes (JSON control messages + base64 or binary audio frames); if Sarvam's actual wire format differs, only `stt/sarvam_stt_client.py` / `tts/sarvam_tts_client.py` need adjusting — the rest of the pipeline is provider-agnostic.
- **ElevenLabs full retirement vs fallback** — see §0.1. Proceeding as full swap.
- **Barge-in (user speaks while TTS is playing)** — not built this phase. Frontend button is a simple record → (auto-stops on silence or manual stop) → listen state machine, not a full-duplex conversation. Flagged as a possible future feature, not started.
- **Audio-output guardrails** — still deferred, per original Decision #4.

## 4. File system layout

```
project/
├── CLAUDE.md
├── .env.example
├── .gitignore
├── requirements.txt
│
├── config/
│   └── settings.py                  # Cell 1 constants + env loading + Sarvam/voice settings (NEW fields)
│
├── ingestion/                       # unchanged, see §1 Job A
│   └── ...
│
├── retrieval/                       # unchanged, see §1 Job B
│   └── ...
│
├── guardrails/                      # unchanged
│   └── ...
│
├── llm_fallback/
│   └── groq_client.py               # unchanged
│
├── tts/
│   ├── text_trim.py                 # unchanged — trim_to_word_limit
│   ├── elevenlabs_client.py         # RETIRED — kept for reference, not called from api/ (see §0.1)
│   └── sarvam_tts_client.py         # NEW — realtime TTS over websocket, streams audio chunks out
│
├── stt/
│   └── sarvam_stt_client.py         # NEW — realtime STT over websocket, language auto-detect,
│                                     #       yields partial + final transcripts
│
├── logging_system/
│   └── logger.py                    # unchanged, now also logs STT/TTS lifecycle events
│
├── timing/
│   └── stage_timer.py               # EXTENDED — adds STT/TTS sub-stage timers (see §7)
│
├── api/
│   ├── app.py                       # existing POST /query (text-in), kept for debugging
│   └── ws_voice.py                  # NEW — WS /voice: browser ⇄ backend ⇄ Sarvam (STT then TTS legs),
│                                     #       runs guardrails → retrieval → gate → groq/trim in between
│
├── frontend/                        # NEW — minimal React voice UI
│   ├── package.json
│   └── src/
│       ├── App.jsx
│       └── VoiceAssistant.jsx       # record/pause/listen button, opens WS /voice, plays streamed audio
│
├── scripts/
│   └── run_ingestion.py             # unchanged
│
└── tests/
```

## 5. Guardrails — unchanged this phase

See prior decisions (#4, #7). Still text-input only, still HTTP-404-style violation reporting (now surfaced as a `{"type": "error", ...}` websocket message instead of an HTTP status, since the voice endpoint is a websocket — see §9).

## 6. Logging system — extended this phase

In addition to the prior requirements, `logging_system/logger.py` now also logs:
- STT session lifecycle: connect, language detected, each partial transcript (debug level), final transcript (info level), disconnect/error
- TTS session lifecycle: connect, first audio chunk sent to client, last audio chunk, disconnect/error
- Any Sarvam websocket error/reconnect

## 7. Timing system — extended this phase (`timing/stage_timer.py`, wired in `api/ws_voice.py`)

Every voice turn returns, over the websocket, a final JSON message shaped like:

```json
{
  "type": "final",
  "transcript": "...",
  "answer_text": "...",
  "timings": {
    "stt_time_to_first_partial_ms": ...,
    "stt_time_to_final_ms": ...,
    "guardrail_check_ms": ...,
    "retrieval_embedding_ms": ...,
    "retrieval_qdrant_ms": ...,
    "retrieval_fusion_ms": ...,
    "retrieval_total_ms": ...,
    "groq_call_ms": null,
    "text_trim_ms": null,
    "tts_time_to_first_chunk_ms": ...,
    "tts_time_to_last_chunk_ms": ...,
    "stt_to_tts_gap_ms": ...,         // NEW: final-transcript timestamp → first-audio-chunk timestamp;
                                       // this is the number the user asked for explicitly ("time between
                                       // STT end to TTS start")
    "total_turn_ms": ...              // mic-stop → last audio chunk sent, true wall-clock
  }
}
```

`TimingCollector` gains two new stage helpers: `.mark("stt_final")` / `.mark("tts_first_chunk")` style timestamp markers (not just start/stop spans), because STT and TTS are streaming and the interesting numbers are gaps *between* events, not just single-stage durations. `stt_to_tts_gap_ms` is computed from these two marks and is the primary "dead air" metric.

## 8. Environment variables (`.env`)

Existing vars (ingestion/retrieval/guardrails/groq/logging) unchanged — see prior version. New this phase:

```
# --- Sarvam AI STT (realtime, websocket) ---
SARVAM_API_KEY=
SARVAM_STT_WS_URL=wss://api.sarvam.ai/speech-to-text/ws   # placeholder — confirm exact path/version in Sarvam docs
SARVAM_STT_LANGUAGE_CODE=unknown                          # "unknown" = auto-detect
SARVAM_STT_SAMPLE_RATE=16000
SARVAM_STT_MODEL=saarika:v2                                # placeholder — confirm current model name

# --- Sarvam AI TTS (realtime, websocket) ---
SARVAM_TTS_WS_URL=wss://api.sarvam.ai/text-to-speech/ws    # placeholder — confirm exact path/version in Sarvam docs
SARVAM_TTS_SPEAKER=meera                                    # placeholder — confirm current speaker/voice name
SARVAM_TTS_SAMPLE_RATE=16000
SARVAM_TTS_LANGUAGE_CODE=en-IN

# --- Legacy TTS (retired, unused, kept for reference — see Decision §0.1) ---
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=

# --- Voice endpoint ---
VOICE_WS_MAX_TURN_SECONDS=30       # safety cutoff for a single recording
```

See `.env.example` in the repo root for the authoritative, always-current copy.

## 9. Voice pipeline flow (`WS /voice`) — this phase's core deliverable

```
Browser (React)                Backend (FastAPI ws_voice.py)              Sarvam AI
     |  click "record"                  |                                      |
     |--- open WS /voice -------------->|                                      |
     |                                  |--- open WS to Sarvam STT ----------->|
     |--- stream mic audio frames ----->|--- relay audio frames -------------->|
     |                                  |<-- partial transcripts --------------|
     |<-- relay partials (optional) ----|                                      |
     |  click "stop" / silence detected |                                      |
     |--- send "stop" control msg ----->|--- signal end-of-audio to Sarvam --->|
     |                                  |<-- final transcript ------------------|
     |                                  |  mark: stt_final
     |                                  |  run_guardrails_strict(transcript)
     |                                  |   -> fail: send {"type":"error",...}, close leg, stop
     |                                  |   -> pass: continue
     |                                  |  retrieve_top_k_hybrid_weighted(...)
     |                                  |  if top1_score < 0.85: call_groq(...)
     |                                  |  else: trim_to_word_limit(...)
     |                                  |--- open WS to Sarvam TTS ----------->|
     |                                  |--- send answer_text ---------------->|
     |                                  |<-- audio chunk 1 ---------------------|
     |                                  |  mark: tts_first_chunk
     |<-- relay audio chunk 1 ----------|                                      |
     |  (start playback immediately)    |<-- audio chunk N ---------------------|
     |<-- relay audio chunk N ----------|                                      |
     |                                  |<-- end-of-audio signal ---------------|
     |                                  |  mark: tts_last_chunk
     |<-- {"type":"final", timings} ----|                                      |
     |  playback finishes, UI resets    |                                      |
```

Key implementation notes:
- Backend never buffers the *whole* TTS output before forwarding — chunks are relayed as they arrive, to keep `stt_to_tts_gap_ms`/perceived latency low.
- Guardrail failure short-circuits before either Sarvam leg for TTS opens — no wasted TTS call on a blocked query.
- Same `hybrid_weighted` retrieval, same 0.85 confidence gate, same Groq/trim branch as the existing text pipeline — `ws_voice.py` is orchestration only, it imports and calls the existing `retrieval/`, `llm_fallback/`, `tts/text_trim.py` modules rather than reimplementing anything (consistent with §0's no-new-logic rule for retrieval, and general "don't duplicate" hygiene for everything else).

## 10. React voice UI (`frontend/src/VoiceAssistant.jsx`) — this phase's core deliverable

- Single component, three visual states: **idle** (mic button), **recording** (button becomes a pause/stop icon, waveform or simple pulsing indicator), **speaking** (button shows a listening/playback indicator while TTS audio plays).
- Mic capture via `MediaRecorder`/`AudioWorklet` → raw PCM frames sent over the `WS /voice` connection as binary websocket messages.
- Playback: incoming binary TTS chunks are queued and played back-to-back via the Web Audio API as they arrive (streaming playback, not "wait for all chunks then play").
- No text chat UI — voice-only, per user's ask ("modern website pause and listen button"). A minimal transcript/answer text display underneath the button is included for debugging/visibility, but is not the focus.
- No external state management library — local component state only, this is a single-page single-component UI.

## 11. Build order (phases)

Prior phases 1–11: ✅ (see previous version of this doc).

**This phase:**
12. ⬜ `stt/sarvam_stt_client.py` — Sarvam STT websocket client (connect, stream audio, yield partial/final transcripts, language detection passthrough)
13. ⬜ `tts/sarvam_tts_client.py` — Sarvam TTS websocket client (connect, send text, yield audio chunks)
14. ⬜ `timing/stage_timer.py` — add mark()/gap timing support
15. ⬜ `api/ws_voice.py` — `WS /voice` orchestrator wiring STT → guardrails → retrieval → gate → groq/trim → TTS
16. ⬜ `frontend/` — React voice UI (`VoiceAssistant.jsx`)
17. ⬜ `.env.example` — add Sarvam + voice vars
18. ⬜ Audio-output guardrails — still deferred, not part of this phase either

**Build sequence for this session, per user instruction: start from STT.** Order is 12 → 14 (timing hooks needed by STT logging) → 13 → 15 → 16.

## 12. What NOT to do

- Do not invent new chunking/scoring/fusion logic not present in the notebook (unchanged rule, retrieval/ingestion only).
- Do not add Odia/Telugu or any multi-language ingestion hooks — English only, per user decision (this is about the *corpus*; STT language auto-detection is a separate, already-approved feature and does not reopen this).
- Do not build audio-output guardrails yet.
- Do not create a `benchmarks/` folder or port Cells 19–50 wholesale.
- Do not silently pick Sarvam model/voice/URL specifics that aren't confirmed from Sarvam's docs — placeholders are marked as such in `.env.example`, fill in for real before deploying.
- Do not batch/buffer full TTS audio before sending to the client — stream chunk-by-chunk (this defeats the point of using a realtime API).
- Do not build barge-in / full-duplex conversation this phase — flagged as future work only.