from __future__ import annotations

import logging
import time

from fastapi import APIRouter
from pydantic import BaseModel

from project.config.settings import settings

from project.guardrails.input_guardrails import (
    run_guardrails_strict,
    GuardrailViolation,
)

from retrieval.factory import retrieve

from project.llm_fallback.groq_client import call_groq

from project.tts.text_trim import trim_to_word_limit

from project.translation.sarvam_translate_client import (
    translate_to_english,
    SarvamTranslationError,
)


logger = logging.getLogger("api.text")

router = APIRouter()


# ============================================================
# REQUEST MODEL
# ============================================================

class TextQueryRequest(BaseModel):
    text: str
    language_code: str = "en-IN"


# ============================================================
# RESPONSE HELPERS
# ============================================================

def _error_response(
    error: str,
    detail: str,
    timings: dict | None = None,
):
    return {
        "type": "error",
        "error": error,
        "detail": detail,
        "timings": timings or {},
    }


# ============================================================
# TEXT ENDPOINT
# ============================================================

@router.post("/text")
async def text_endpoint(request: TextQueryRequest):

    turn_start = time.perf_counter()

    original_text = (request.text or "").strip()
    language_code = (
        request.language_code or "en-IN"
    ).strip()

    logger.info(
        "text.turn_start language=%s chars=%d",
        language_code,
        len(original_text),
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not original_text:
        return _error_response(
            "invalid_request",
            "Text cannot be empty.",
        )

    # ========================================================
    # 1. GUARDRAILS
    # ========================================================

    guardrail_start = time.perf_counter()

    try:
        run_guardrails_strict(original_text)

    except GuardrailViolation as exc:

        guardrail_ms = (
            time.perf_counter()
            - guardrail_start
        ) * 1000.0

        logger.info(
            "text.guardrail_blocked "
            "duration_ms=%.2f "
            "check=%s",
            guardrail_ms,
            getattr(exc, "check", "unknown"),
        )

        return _error_response(
            "guardrail_violation",
            str(exc),
            {
                "guardrail_ms": round(
                    guardrail_ms,
                    2,
                )
            },
        )

    guardrail_ms = (
        time.perf_counter()
        - guardrail_start
    ) * 1000.0

    logger.info(
        "text.stage_complete "
        "stage=guardrails "
        "duration_ms=%.2f",
        guardrail_ms,
    )

    # ========================================================
    # 2. TRANSLATE USER QUESTION → ENGLISH
    # ========================================================

    translation_start = time.perf_counter()

    try:

        translation_result = await translate_to_english(
            text=original_text,
            source_language_code=language_code,
        )

        retrieval_query = (
            translation_result.translated_text
        )

        translation_ms = (
            translation_result.translation_ms
        )

    except SarvamTranslationError as exc:

        translation_ms = (
            time.perf_counter()
            - translation_start
        ) * 1000.0

        logger.error(
            "text.translation_failed "
            "duration_ms=%.2f "
            "detail=%s",
            translation_ms,
            exc,
        )

        return _error_response(
            "translation_failed",
            str(exc),
            {
                "guardrail_ms": round(
                    guardrail_ms,
                    2,
                ),
                "translation_to_english_ms": round(
                    translation_ms,
                    2,
                ),
            },
        )

    logger.info(
        "text.translation_complete "
        "source=%s "
        "target=en-IN "
        "duration_ms=%.2f "
        "original=%r "
        "retrieval_query=%r",
        language_code,
        translation_ms,
        original_text,
        retrieval_query,
    )

    # ========================================================
    # 3. RETRIEVAL
    # ========================================================

    retrieval_start = time.perf_counter()

    try:

        retrieval_result = retrieve(
            retrieval_query,
            mode=settings.RETRIEVAL_MODE,
        )

    except Exception as exc:

        retrieval_wall_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000.0

        logger.exception(
            "text.retrieval_failed "
            "duration_ms=%.2f",
            retrieval_wall_ms,
        )

        return _error_response(
            "retrieval_failed",
            str(exc),
            {
                "guardrail_ms": round(
                    guardrail_ms,
                    2,
                ),
                "translation_to_english_ms": round(
                    translation_ms,
                    2,
                ),
                "retrieval_wall_ms": round(
                    retrieval_wall_ms,
                    2,
                ),
            },
        )

    retrieval_wall_ms = (
        time.perf_counter()
        - retrieval_start
    ) * 1000.0

    results = retrieval_result.get(
        "results",
        [],
    )

    embedding_ms = retrieval_result.get(
        "embedding_ms"
    )

    qdrant_ms = retrieval_result.get(
        "qdrant_ms"
    )

    fusion_ms = retrieval_result.get(
        "fusion_ms"
    )

    retrieval_total_ms = retrieval_result.get(
        "total_ms"
    )

    if results:
        top1_score = float(
            results[0].score
        )
    else:
        top1_score = 0.0

    logger.info(
        "text.stage_complete "
        "stage=retrieval "
        "wall_ms=%.2f "
        "embedding_ms=%s "
        "qdrant_ms=%s "
        "fusion_ms=%s "
        "retrieval_total_ms=%s "
        "top1_score=%.4f "
        "n_results=%d",
        retrieval_wall_ms,
        embedding_ms,
        qdrant_ms,
        fusion_ms,
        retrieval_total_ms,
        top1_score,
        len(results),
    )

    # ========================================================
    # 4. CONFIDENCE GATE
    # ========================================================

    llm_used = False
    generation_method = "retrieval"

    groq_ms = 0.0

    if (
        results
        and top1_score
        >= settings.CONFIDENCE_THRESHOLD
    ):

        logger.info(
            "text.confidence_gate "
            "decision=HIGH_CONFIDENCE "
            "score=%.4f "
            "threshold=%.4f "
            "llm_used=false",
            top1_score,
            settings.CONFIDENCE_THRESHOLD,
        )

        # ----------------------------------------------------
        # High confidence
        # Use retrieved English text directly
        # ----------------------------------------------------

        source_text = (
            results[0]
            .payload
            .get("text_en", "")
        )

        trim_result = trim_to_word_limit(
            source_text,
            settings.TTS_WORD_LIMIT,
        )

        if isinstance(
            trim_result,
            dict,
        ):
            answer_english = trim_result[
                "text"
            ]
        else:
            answer_english = trim_result

    else:

        # ----------------------------------------------------
        # Low confidence
        # Use Groq
        # ----------------------------------------------------

        llm_used = True
        generation_method = "groq"

        logger.info(
            "text.confidence_gate "
            "decision=LOW_CONFIDENCE "
            "score=%.4f "
            "threshold=%.4f "
            "llm_used=true",
            top1_score,
            settings.CONFIDENCE_THRESHOLD,
        )

        groq_start = time.perf_counter()

        try:

            groq_result = call_groq(
                retrieval_query,
                results,
            )

        except Exception as exc:

            groq_ms = (
                time.perf_counter()
                - groq_start
            ) * 1000.0

            logger.exception(
                "text.groq_failed "
                "duration_ms=%.2f",
                groq_ms,
            )

            return _error_response(
                "generation_failed",
                str(exc),
                {
                    "guardrail_ms": round(
                        guardrail_ms,
                        2,
                    ),
                    "translation_to_english_ms": round(
                        translation_ms,
                        2,
                    ),
                    "retrieval_wall_ms": round(
                        retrieval_wall_ms,
                        2,
                    ),
                    "groq_ms": round(
                        groq_ms,
                        2,
                    ),
                },
            )

        groq_ms = (
            time.perf_counter()
            - groq_start
        ) * 1000.0

        if isinstance(
            groq_result,
            dict,
        ):
            answer_english = (
                groq_result["answer"]
            )
        else:
            answer_english = groq_result

    answer_english = (
        answer_english or ""
    ).strip()

    logger.info(
        "text.answer_ready "
        "method=%s "
        "llm_used=%s "
        "chars=%d",
        generation_method,
        llm_used,
        len(answer_english),
    )

    # ========================================================
    # 5. TRANSLATE ANSWER → USER LANGUAGE
    # ========================================================
    #
    # IMPORTANT:
    #
    # English input:
    #     No translation.
    #
    # Hindi input:
    #     English answer → Hindi
    #
    # Odia input:
    #     English answer → Odia
    #
    # Tamil input:
    #     English answer → Tamil
    #
    # ========================================================

    answer_translation_ms = 0.0
    final_answer = answer_english

    if language_code != "en-IN":

        logger.info(
            "text.answer_translation_start "
            "target=%s chars=%d",
            language_code,
            len(answer_english),
        )

        answer_translation_start = (
            time.perf_counter()
        )

        try:

            # ------------------------------------------------
            # IMPORTANT:
            #
            # This uses Sarvam's translation endpoint directly.
            #
            # If your translation client already has a
            # translate_answer_to_user_language() function,
            # replace this block with that function.
            # ------------------------------------------------

            from project.translation.sarvam_translate_client import (
                translate_text,
            )

            answer_translation_result = (
                await translate_text(
                    text=answer_english,
                    source_language_code="en-IN",
                    target_language_code=language_code,
                )
            )

            final_answer = (
                answer_translation_result
                .translated_text
            )

            answer_translation_ms = (
                answer_translation_result
                .translation_ms
            )

        except ImportError:

            logger.exception(
                "text.answer_translation_failed "
                "reason=translate_text_not_available"
            )

            return _error_response(
                "answer_translation_not_available",
                (
                    "translate_text() is not "
                    "available in "
                    "sarvam_translate_client.py"
                ),
            )

        except SarvamTranslationError as exc:

            answer_translation_ms = (
                time.perf_counter()
                - answer_translation_start
            ) * 1000.0

            logger.error(
                "text.answer_translation_failed "
                "target=%s "
                "duration_ms=%.2f "
                "detail=%s",
                language_code,
                answer_translation_ms,
                exc,
            )

            return _error_response(
                "answer_translation_failed",
                str(exc),
                {
                    "answer_translation_ms": round(
                        answer_translation_ms,
                        2,
                    )
                },
            )

        logger.info(
            "text.answer_translation_complete "
            "target=%s "
            "duration_ms=%.2f "
            "output_chars=%d",
            language_code,
            answer_translation_ms,
            len(final_answer),
        )

    else:

        logger.info(
            "text.answer_translation_skipped "
            "reason=already_english"
        )

    # ========================================================
    # 6. TOTAL TIMING
    # ========================================================

    total_ms = (
        time.perf_counter()
        - turn_start
    ) * 1000.0

    timings = {
        "guardrail_ms": round(
            guardrail_ms,
            2,
        ),

        "translation_to_english_ms": round(
            translation_ms,
            2,
        ),

        "retrieval_wall_ms": round(
            retrieval_wall_ms,
            2,
        ),

        "embedding_ms": round(
            float(embedding_ms or 0.0),
            2,
        ),

        "qdrant_ms": round(
            float(qdrant_ms or 0.0),
            2,
        ),

        "fusion_ms": round(
            float(fusion_ms or 0.0),
            2,
        ),

        "retrieval_engine_ms": round(
            float(
                retrieval_total_ms or 0.0
            ),
            2,
        ),

        "groq_ms": round(
            groq_ms,
            2,
        ),

        "answer_translation_ms": round(
            answer_translation_ms,
            2,
        ),

        "total_ms": round(
            total_ms,
            2,
        ),
    }

    logger.info(
        "text.timing_summary "
        "language=%s "
        "translation_ms=%.2f "
        "guardrail_ms=%.2f "
        "retrieval_wall_ms=%.2f "
        "groq_ms=%.2f "
        "answer_translation_ms=%.2f "
        "total_ms=%.2f",
        language_code,
        translation_ms,
        guardrail_ms,
        retrieval_wall_ms,
        groq_ms,
        answer_translation_ms,
        total_ms,
    )

    logger.info(
        "text.turn_complete "
        "language=%s "
        "generation=%s "
        "llm_used=%s "
        "top1_score=%.4f "
        "total_ms=%.2f",
        language_code,
        generation_method,
        llm_used,
        top1_score,
        total_ms,
    )

    # ========================================================
    # 7. RESPONSE
    # ========================================================

    return {
        "type": "text_response",

        # Original user question
        "text": original_text,

        # English question used for Qdrant
        "retrieval_query": retrieval_query,

        # Final answer in user's language
        "answer": final_answer,

        "language_code": language_code,

        "generation_method": generation_method,

        "llm_used": llm_used,

        "top1_score": top1_score,

        "results_count": len(results),

        "timings": timings,
    }