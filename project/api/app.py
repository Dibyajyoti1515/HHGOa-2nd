"""
api/app.py

FastAPI serving layer. NEW module -- not from the notebook. Wires
together everything built so far into one request flow, and exposes
it under a versioned /v1 prefix so the frontend only ever talks to
/v1/... (both the text POST /query path and the WS /voice path from
api/ws_voice.py).

    guardrails
      -> retrieval (settings.RETRIEVAL_MODE, default hybrid_weighted
         from Cell 39)
      -> confidence gate (settings.CONFIDENCE_THRESHOLD, CLAUDE.md
         decision #3):
             top1_score <  threshold -> Groq composes an answer from
                                         the retrieved context
             top1_score >= threshold -> sentence-boundary trim of the
                                         raw retrieved text (no LLM)
      -> ElevenLabs TTS
      -> { "audio_url": ..., "timings": {...} }

On any guardrail failure: HTTP 404 with the guardrail's error body
(CLAUDE.md decision #4).

Timing note: retrieval measures dense+sparse embedding together as a
single embedding_ms (that's how the ported Cell 39 function itself
measures it -- t0 before both encode calls, t1 after), so this API
does not report separate dense/sparse embedding times. Splitting that
would mean editing the ported retrieval function, which is off-limits
per CLAUDE.md's "no new retrieval logic" rule.

Versioning note (this change): all routes previously registered
directly on `app` now live on an APIRouter mounted at /v1. This is
routing/orchestration only -- no change to guardrail, retrieval,
confidence-gate, or TTS logic. /audio is intentionally left
unprefixed since it's a static file mount, not an API route the
frontend calls by contract -- only the audio_url value it's handed.
"""

from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from project.config.settings import settings
from project.guardrails.input_guardrails import (
    GuardrailViolation,
    build_error_response,
    run_guardrails_strict,
)
from project.llm_fallback.groq_client import call_groq
from project.logging_system.logger import get_logger
from retrieval.factory import retrieve
from project.timing.stage_timer import TimingCollector
from project.tts.elevenlabs_client import call_elevenlabs_tts
from project.tts.text_trim import trim_to_word_limit

# ws_voice.py exposes `router = APIRouter()` with the `WS /voice` route
# (see CLAUDE.md §9 / §11). We mount it under the same /v1 prefix here
# rather than duplicating anything from that module.
from project.api.ws_voice import router as voice_router
from project.api.text import router as text_router

logger = get_logger(__name__)

settings.ensure_dirs()

app = FastAPI(title="Hybrid Retrieval + Voice Answer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated audio files at /audio/<filename> -- deliberately
# outside /v1, see module docstring.
app.mount("/audio", StaticFiles(directory=settings.TTS_OUTPUT_DIR), name="audio")

# All versioned API routes attach to this router, then get included
# into `app` once at the bottom of the file.
v1_router = APIRouter(prefix="/v1")


class QueryRequest(BaseModel):
    query: str


@v1_router.post("/query")
def query(request: QueryRequest):
    timing = TimingCollector()

    # ---- Guardrails (CLAUDE.md section 5) ----
    try:
        with timing.stage("guardrail_check_ms"):
            run_guardrails_strict(request.query)
    except GuardrailViolation as exc:
        logger.info("Guardrail rejected query=%r check=%s", request.query, exc.check_name)
        return JSONResponse(status_code=404, content=build_error_response(exc))

    # ---- Retrieval (settings.RETRIEVAL_MODE, default hybrid_weighted) ----
    retrieval_result = retrieve(request.query)
    results = retrieval_result["results"]

    timing.record_ms("retrieval_embedding_ms", retrieval_result.get("embedding_ms"))
    timing.record_ms("retrieval_qdrant_ms", retrieval_result.get("qdrant_ms"))
    timing.record_ms("retrieval_fusion_ms", retrieval_result.get("fusion_ms"))
    timing.record_ms("retrieval_total_ms", retrieval_result.get("total_ms"))

    top1_score = results[0].score if results else None
    logger.info(
        "Retrieval done: query=%r mode=%s top1_score=%s n_results=%s",
        request.query, settings.RETRIEVAL_MODE, top1_score, len(results),
    )

    # ---- Confidence gate (CLAUDE.md decision #3) ----
    if top1_score is not None and top1_score >= settings.CONFIDENCE_THRESHOLD:
        # High confidence: skip the LLM, just trim the retrieved text.
        source_text = results[0].payload.get("text_en", "")
        trim_result = trim_to_word_limit(source_text)
        final_text = trim_result["text"]
        timing.record_ms("groq_call_ms", None)
        timing.record_ms("text_trim_ms", trim_result["trim_ms"])
        logger.info("Confidence gate: skipped LLM (top1_score=%.4f)", top1_score)
    else:
        # Low/no confidence: let Groq compose an answer from retrieved context.
        groq_result = call_groq(request.query, results)
        final_text = groq_result["answer"]
        timing.record_ms("groq_call_ms", groq_result["groq_ms"])
        timing.record_ms("text_trim_ms", None)
        logger.info("Confidence gate: used Groq fallback (top1_score=%s)", top1_score)

    # ---- TTS ----
    tts_result = call_elevenlabs_tts(final_text)

    audio_filename = Path(tts_result["audio_path"]).name
    audio_url = f"/audio/{audio_filename}"

    return {
        "audio_url": audio_url,
        "timings": timing.as_dict(),
    }


@v1_router.get("/health")
def health():
    return {"status": "ok"}


# ---- Mount versioned routers onto the app ----
# POST /v1/query, GET /v1/health
app.include_router(v1_router)
# WS /v1/voice (route itself defined in api/ws_voice.py as "/voice";
# the /v1 prefix is applied here, not duplicated in that file)
app.include_router(voice_router, prefix="/v1")
app.include_router(text_router, prefix="/v1")