"""
api/ws_voice.py

Voice WebSocket orchestration.

Flow:

    Browser audio
        ↓
    Sarvam STT
        ↓
    Original transcript + detected language
        ↓
    Sarvam Translate: user language -> English
        ↓
    English retrieval query
        ↓
    Guardrails
        ↓
    Hybrid retrieval
        ↓
    Confidence gate
        ↓
    ┌──────────────────────────────┐
    │ High confidence             │
    │ Retrieved text -> answer    │
    │                              │
    │ Low confidence              │
    │ Groq -> English answer      │
    └──────────────────────────────┘
        ↓
    Sarvam Translate:
        English answer -> user's language
        ↓
    Sarvam TTS using original language
        ↓
    Browser audio
        ↓
    Final JSON + timings
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from project.config.settings import settings

from project.guardrails.input_guardrails import (
    run_guardrails_strict,
    GuardrailViolation,
)

from retrieval.factory import retrieve

from project.llm_fallback.groq_client import call_groq

from project.tts.text_trim import trim_to_word_limit

from project.stt.sarvam_stt_client import (
    transcribe_stream,
    SarvamSTTError,
)

from project.tts.sarvam_tts_client import (
    synthesize_stream,
    SarvamTTSError,
)

from project.translation.sarvam_translate_client import (
    translate_to_english,
    translate_answer_to_user_language,
    SarvamTranslationError,
)

from project.timing.stage_timer import TimingCollector


logger = logging.getLogger("api.ws_voice")

router = APIRouter()


# ============================================================
# Browser audio
# ============================================================


async def _browser_audio_frames(
    ws: WebSocket,
    stop_event: asyncio.Event,
):
    """
    Receive raw PCM audio frames from the browser.

    Stops when:
      - browser sends {"event": "stop"}
      - websocket disconnects
      - VOICE_WS_MAX_TURN_SECONDS is reached
    """

    frame_count = 0
    total_bytes = 0

    async def _watchdog():
        await asyncio.sleep(
            settings.VOICE_WS_MAX_TURN_SECONDS
        )

        logger.warning(
            "voice.turn_timeout_hit max_seconds=%s",
            settings.VOICE_WS_MAX_TURN_SECONDS,
        )

        stop_event.set()

    watchdog_task = asyncio.create_task(_watchdog())

    try:
        while not stop_event.is_set():

            message = await ws.receive()

            # ------------------------------------------------
            # WebSocket disconnected
            # ------------------------------------------------

            if message.get("type") == "websocket.disconnect":
                logger.info(
                    "voice.browser_disconnected"
                )

                stop_event.set()
                break

            # ------------------------------------------------
            # Binary audio
            # ------------------------------------------------

            if (
                "bytes" in message
                and message["bytes"] is not None
            ):
                audio = message["bytes"]

                frame_count += 1
                total_bytes += len(audio)

                logger.info(
                    "voice.browser_audio "
                    "frame=%d "
                    "bytes=%d "
                    "total_bytes=%d",
                    frame_count,
                    len(audio),
                    total_bytes,
                )

                yield audio

            # ------------------------------------------------
            # Text/control message
            # ------------------------------------------------

            elif (
                "text" in message
                and message["text"] is not None
            ):
                text_message = message["text"]

                logger.info(
                    "voice.browser_control message=%s",
                    text_message,
                )

                if '"stop"' in text_message:
                    logger.info(
                        "voice.browser_stop_received"
                    )

                    stop_event.set()
                    break

    finally:

        logger.info(
            "voice.browser_audio_finished "
            "frames=%d "
            "total_bytes=%d",
            frame_count,
            total_bytes,
        )

        watchdog_task.cancel()


# ============================================================
# Stage logging helpers
# ============================================================


def _stage_start(
    stage: str,
    **kwargs,
):
    """
    Log stage start.
    """

    if kwargs:
        extra = " ".join(
            f"{key}={value}"
            for key, value in kwargs.items()
        )

        logger.info(
            "voice.stage_start stage=%s %s",
            stage,
            extra,
        )
    else:
        logger.info(
            "voice.stage_start stage=%s",
            stage,
        )


def _stage_complete(
    stage: str,
    duration_ms: float,
    **kwargs,
):
    """
    Log stage completion.
    """

    if kwargs:
        extra = " ".join(
            f"{key}={value}"
            for key, value in kwargs.items()
        )

        logger.info(
            "voice.stage_complete "
            "stage=%s "
            "duration_ms=%.2f "
            "%s",
            stage,
            duration_ms,
            extra,
        )

    else:
        logger.info(
            "voice.stage_complete "
            "stage=%s "
            "duration_ms=%.2f",
            stage,
            duration_ms,
        )


# ============================================================
# Main voice endpoint
# ============================================================


@router.websocket("/voice")
async def voice_endpoint(ws: WebSocket):

    await ws.accept()

    timing = TimingCollector()

    stop_event = asyncio.Event()

    turn_start = time.perf_counter()

    logger.info("voice.turn_start")

    try:

        # ====================================================
        # 1. STT
        # ====================================================

        _stage_start("stt")

        stt_start = time.perf_counter()

        try:

            stt_result = await transcribe_stream(
                audio_frames=_browser_audio_frames(
                    ws,
                    stop_event,
                ),
                timing=timing,
                stop_event=stop_event,
            )

        except SarvamSTTError as exc:

            stt_ms = (
                time.perf_counter()
                - stt_start
            ) * 1000.0

            _stage_complete(
                "stt",
                stt_ms,
                status="failed",
            )

            logger.error(
                "voice.stt_failed "
                "duration_ms=%.2f "
                "detail=%s",
                stt_ms,
                exc,
            )

            await ws.send_json(
                {
                    "type": "error",
                    "stage": "stt",
                    "detail": str(exc),
                }
            )

            await ws.close()

            return

        stt_ms = (
            time.perf_counter()
            - stt_start
        ) * 1000.0

        transcript = (
            stt_result.transcript or ""
        ).strip()

        language_code = (
            stt_result.language_code
            or "en-IN"
        )

        _stage_complete(
            "stt",
            stt_ms,
            language=language_code,
            text_len=len(transcript),
        )

        logger.info(
            "voice.transcript "
            "language=%s "
            "text_len=%d "
            "text=%r",
            language_code,
            len(transcript),
            transcript,
        )

        # Send original transcript to frontend.
        await ws.send_json(
            {
                "type": "transcript",

                # Original user speech transcript
                "text": transcript,

                # Detected language
                "language": language_code,

                # Useful for frontend
                "language_code": language_code,
            }
        )

        # ====================================================
        # 2. TRANSLATE USER LANGUAGE -> ENGLISH
        # ====================================================

        _stage_start(
            "translation_to_english",
            source_language=language_code,
            target_language="en-IN",
        )

        translation_start = time.perf_counter()

        try:

            translation_result = (
                await translate_to_english(
                    text=transcript,
                    source_language_code=language_code,
                )
            )

        except SarvamTranslationError as exc:

            translation_ms = (
                time.perf_counter()
                - translation_start
            ) * 1000.0

            _stage_complete(
                "translation_to_english",
                translation_ms,
                status="failed",
            )

            logger.error(
                "voice.translation_to_english_failed "
                "duration_ms=%.2f "
                "detail=%s",
                translation_ms,
                exc,
            )

            await ws.send_json(
                {
                    "type": "error",
                    "stage": "translation_to_english",
                    "detail": str(exc),
                }
            )

            await ws.close()

            return

        translation_to_english_ms = (
            time.perf_counter()
            - translation_start
        ) * 1000.0

        retrieval_query = (
            translation_result.translated_text
            or transcript
        ).strip()

        timing.record_ms(
            "translation_to_english",
            translation_to_english_ms,
        )

        _stage_complete(
            "translation_to_english",
            translation_to_english_ms,
            source_language=language_code,
            target_language="en-IN",
            chars=len(retrieval_query),
        )

        logger.info(
            "voice.translation_to_english_complete "
            "source=%s "
            "target=en-IN "
            "duration_ms=%.2f "
            "retrieval_query=%r",
            language_code,
            translation_to_english_ms,
            retrieval_query,
        )

        # Send retrieval query to frontend optionally.
        await ws.send_json(
            {
                "type": "translation",
                "direction": "to_english",
                "source_language": language_code,
                "target_language": "en-IN",
                "text": retrieval_query,
                "duration_ms": round(
                    translation_to_english_ms,
                    2,
                ),
            }
        )

        # ====================================================
        # 3. GUARDRAILS
        # ====================================================

        _stage_start("guardrails")

        guardrail_start = time.perf_counter()

        try:

            # IMPORTANT:
            # Guardrails operate on the original
            # user transcript.

            run_guardrails_strict(
                transcript
            )

        except GuardrailViolation as exc:

            guardrail_ms = (
                time.perf_counter()
                - guardrail_start
            ) * 1000.0

            _stage_complete(
                "guardrails",
                guardrail_ms,
                status="blocked",
            )

            logger.info(
                "voice.guardrail_blocked "
                "check=%s",
                getattr(
                    exc,
                    "check",
                    "unknown",
                ),
            )

            await ws.send_json(
                {
                    "type": "error",
                    "stage": "guardrails",
                    "error": "guardrail_violation",
                    **_guardrail_detail(exc),
                }
            )

            await ws.close()

            return

        guardrail_ms = (
            time.perf_counter()
            - guardrail_start
        ) * 1000.0

        timing.record_ms(
            "guardrails",
            guardrail_ms,
        )

        _stage_complete(
            "guardrails",
            guardrail_ms,
        )

        # ====================================================
        # 4. RETRIEVAL
        # ====================================================

        _stage_start(
            "retrieval",
            mode=settings.RETRIEVAL_MODE,
        )

        retrieval_wall_start = (
            time.perf_counter()
        )

        retrieval_result = retrieve(
            retrieval_query,
            mode=settings.RETRIEVAL_MODE,
        )

        retrieval_wall_ms = (
            time.perf_counter()
            - retrieval_wall_start
        ) * 1000.0

        results = (
            retrieval_result.get(
                "results",
                [],
            )
        )

        embedding_ms = (
            retrieval_result.get(
                "embedding_ms"
            )
        )

        qdrant_ms = (
            retrieval_result.get(
                "qdrant_ms"
            )
        )

        fusion_ms = (
            retrieval_result.get(
                "fusion_ms"
            )
        )

        retrieval_total_ms = (
            retrieval_result.get(
                "total_ms"
            )
        )

        timing.record_ms(
            "retrieval_wall",
            retrieval_wall_ms,
        )

        timing.record_ms(
            "retrieval_embedding",
            embedding_ms,
        )

        timing.record_ms(
            "retrieval_qdrant",
            qdrant_ms,
        )

        timing.record_ms(
            "retrieval_fusion",
            fusion_ms,
        )

        timing.record_ms(
            "retrieval_total",
            retrieval_total_ms,
        )

        top1_score = (
            results[0].score
            if results
            else None
        )

        _stage_complete(
            "retrieval",
            retrieval_wall_ms,
            embedding_ms=embedding_ms,
            qdrant_ms=qdrant_ms,
            fusion_ms=fusion_ms,
            retrieval_total_ms=retrieval_total_ms,
            top1_score=top1_score,
            n_results=len(results),
        )

        logger.info(
            "voice.retrieval_done "
            "query=%r "
            "top1_score=%s "
            "n_results=%d",
            retrieval_query,
            top1_score,
            len(results),
        )

        # ====================================================
        # 5. CONFIDENCE GATE
        # ====================================================

        llm_used = False

        generation_method = "retrieval"

        # ----------------------------------------------------
        # High confidence
        # ----------------------------------------------------

        if (
            top1_score is not None
            and top1_score
            >= settings.CONFIDENCE_THRESHOLD
        ):

            logger.info(
                "voice.confidence_gate "
                "decision=HIGH_CONFIDENCE "
                "score=%s "
                "threshold=%s "
                "llm_used=false",
                top1_score,
                settings.CONFIDENCE_THRESHOLD,
            )

            _stage_start("text_trim")

            trim_start = time.perf_counter()

            source_text = (
                results[0]
                .payload
                .get(
                    "text_en",
                    "",
                )
            )

            trim_result = trim_to_word_limit(
                source_text,
                settings.TTS_WORD_LIMIT,
            )

            answer_text = (
                trim_result["text"]
                if isinstance(
                    trim_result,
                    dict,
                )
                else trim_result
            )

            trim_ms = (
                time.perf_counter()
                - trim_start
            ) * 1000.0

            timing.record_ms(
                "text_trim",
                trim_ms,
            )

            _stage_complete(
                "text_trim",
                trim_ms,
                chars=len(answer_text),
            )

            generation_method = (
                "retrieval_trim"
            )

        # ----------------------------------------------------
        # Low confidence
        # ----------------------------------------------------

        else:

            llm_used = True

            generation_method = "groq"

            logger.info(
                "voice.confidence_gate "
                "decision=LOW_CONFIDENCE "
                "score=%s "
                "threshold=%s "
                "llm_used=true",
                top1_score,
                settings.CONFIDENCE_THRESHOLD,
            )

            _stage_start("groq")

            groq_start = time.perf_counter()

            groq_result = call_groq(
                retrieval_query,
                results,
            )

            answer_text = (
                groq_result["answer"]
                if isinstance(
                    groq_result,
                    dict,
                )
                else groq_result
            )

            groq_ms = (
                time.perf_counter()
                - groq_start
            ) * 1000.0

            timing.record_ms(
                "groq",
                groq_ms,
            )

            _stage_complete(
                "groq",
                groq_ms,
                chars=len(answer_text),
            )

        answer_text = (
            answer_text or ""
        ).strip()

        logger.info(
            "voice.answer_ready "
            "method=%s "
            "llm_used=%s "
            "top1_score=%s "
            "chars=%d",
            generation_method,
            llm_used,
            top1_score,
            len(answer_text),
        )

        # ====================================================
        # 6. ENGLISH ANSWER -> USER LANGUAGE
        # ====================================================

        original_english_answer = answer_text

        # English users do not need translation.
        if language_code == "en-IN":

            answer_translation_ms = 0.0

            logger.info(
                "voice.answer_translation_skipped "
                "language=en-IN"
            )

        else:

            _stage_start(
                "translation_to_user_language",
                source_language="en-IN",
                target_language=language_code,
            )

            answer_translation_start = (
                time.perf_counter()
            )

            try:

                answer_translation_result = (
                    await translate_answer_to_user_language(
                        text=answer_text,
                        target_language_code=language_code,
                    )
                )

            except SarvamTranslationError as exc:

                answer_translation_ms = (
                    time.perf_counter()
                    - answer_translation_start
                ) * 1000.0

                _stage_complete(
                    "translation_to_user_language",
                    answer_translation_ms,
                    status="failed",
                )

                logger.error(
                    "voice.answer_translation_failed "
                    "source=en-IN "
                    "target=%s "
                    "duration_ms=%.2f "
                    "detail=%s",
                    language_code,
                    answer_translation_ms,
                    exc,
                )

                await ws.send_json(
                    {
                        "type": "error",
                        "stage": "translation_to_user_language",
                        "detail": str(exc),
                    }
                )

                await ws.close()

                return

            answer_translation_ms = (
                time.perf_counter()
                - answer_translation_start
            ) * 1000.0

            answer_text = (
                answer_translation_result
                .translated_text
                .strip()
            )

            timing.record_ms(
                "translation_to_user_language",
                answer_translation_ms,
            )

            _stage_complete(
                "translation_to_user_language",
                answer_translation_ms,
                source_language="en-IN",
                target_language=language_code,
                chars=len(answer_text),
            )

            logger.info(
                "voice.answer_translated "
                "source=en-IN "
                "target=%s "
                "duration_ms=%.2f "
                "chars=%d",
                language_code,
                answer_translation_ms,
                len(answer_text),
            )

        # ====================================================
        # Send answer text BEFORE TTS
        # ====================================================

        await ws.send_json(
            {
                "type": "answer",
                "text": answer_text,
                "language": language_code,
                "language_code": language_code,
                "generation": generation_method,
                "llm_used": llm_used,
            }
        )

        # ====================================================
        # 7. TTS
        # ====================================================

        _stage_start(
            "tts",
            model="bulbul:v3",
            language=language_code,
            chars=len(answer_text),
        )

        tts_start = time.perf_counter()

        tts_first_chunk_ms = None

        tts_chunk_count = 0

        try:

            async for audio_chunk in synthesize_stream(
                answer_text,

                # IMPORTANT:
                # TTS gets the ORIGINAL detected
                # user language.
                language_code=language_code,

                timing=timing,
            ):

                tts_chunk_count += 1

                if tts_first_chunk_ms is None:

                    tts_first_chunk_ms = (
                        time.perf_counter()
                        - tts_start
                    ) * 1000.0

                    timing.record_ms(
                        "tts_first_chunk",
                        tts_first_chunk_ms,
                    )

                    logger.info(
                        "voice.tts_first_audio "
                        "duration_ms=%.2f "
                        "chunk=%d",
                        tts_first_chunk_ms,
                        tts_chunk_count,
                    )

                await ws.send_bytes(
                    audio_chunk
                )

        except SarvamTTSError as exc:

            tts_ms = (
                time.perf_counter()
                - tts_start
            ) * 1000.0

            timing.record_ms(
                "tts",
                tts_ms,
            )

            _stage_complete(
                "tts",
                tts_ms,
                status="failed",
                chunks=tts_chunk_count,
            )

            logger.error(
                "voice.tts_failed "
                "duration_ms=%.2f "
                "language=%s "
                "detail=%s",
                tts_ms,
                language_code,
                exc,
            )

            await ws.send_json(
                {
                    "type": "error",
                    "stage": "tts",
                    "detail": str(exc),
                }
            )

            await ws.close()

            return

        tts_ms = (
            time.perf_counter()
            - tts_start
        ) * 1000.0

        timing.record_ms(
            "tts",
            tts_ms,
        )

        _stage_complete(
            "tts",
            tts_ms,
            chunks=tts_chunk_count,
            first_chunk_ms=tts_first_chunk_ms,
        )

        # ====================================================
        # 8. TOTAL TIMING
        # ====================================================

        total_turn_ms = (
            time.perf_counter()
            - turn_start
        ) * 1000.0

        # Some TimingCollector implementations
        # calculate this themselves.
        try:
            collector_total_ms = (
                timing.total_turn_ms()
            )
        except Exception:
            collector_total_ms = total_turn_ms

        # ====================================================
        # Timing summary
        # ====================================================

        logger.info(
            "voice.timing_summary "
            "stt_ms=%.2f "
            "translation_to_english_ms=%.2f "
            "guardrails_ms=%.2f "
            "retrieval_wall_ms=%.2f "
            "embedding_ms=%s "
            "qdrant_ms=%s "
            "fusion_ms=%s "
            "retrieval_total_ms=%s "
            "generation=%s "
            "llm_used=%s "
            "groq_ms=%s "
            "translation_to_user_language_ms=%.2f "
            "tts_ms=%.2f "
            "tts_first_chunk_ms=%s "
            "total_turn_ms=%.2f",

            stt_ms,

            translation_to_english_ms,

            timing.get(
                "guardrails",
                0.0,
            ) if hasattr(timing, "get") else 0.0,

            retrieval_wall_ms,

            embedding_ms,

            qdrant_ms,

            fusion_ms,

            retrieval_total_ms,

            generation_method,

            llm_used,

            timing.get(
                "groq",
                0.0,
            ) if hasattr(timing, "get") else None,

            answer_translation_ms,

            tts_ms,

            tts_first_chunk_ms,

            collector_total_ms,
        )

        # ====================================================
        # 9. FINAL MESSAGE
        # ====================================================

        timings_dict = timing.as_dict()

        # Add explicit fields even if TimingCollector
        # implementation changes.
        timings_dict.update(
            {
                "stt_ms": round(
                    stt_ms,
                    2,
                ),

                "translation_to_english_ms": round(
                    translation_to_english_ms,
                    2,
                ),

                "translation_to_user_language_ms": round(
                    answer_translation_ms,
                    2,
                ),

                "retrieval_wall_ms": round(
                    retrieval_wall_ms,
                    2,
                ),

                "embedding_ms": embedding_ms,

                "qdrant_ms": qdrant_ms,

                "fusion_ms": fusion_ms,

                "retrieval_total_ms": retrieval_total_ms,

                "tts_ms": round(
                    tts_ms,
                    2,
                ),

                "tts_first_chunk_ms": (
                    round(
                        tts_first_chunk_ms,
                        2,
                    )
                    if tts_first_chunk_ms
                    is not None
                    else None
                ),

                "total_turn_ms": round(
                    collector_total_ms,
                    2,
                ),
            }
        )

        await ws.send_json(
            {
                "type": "final",

                # Original user question
                "transcript": transcript,

                # English question used for DB
                "retrieval_query": retrieval_query,

                # Final answer in user's language
                "answer_text": answer_text,

                # English answer before final translation
                "english_answer": (
                    original_english_answer
                ),

                # User language
                "language": language_code,
                "language_code": language_code,

                # Generation method
                "generation": generation_method,
                "llm_used": llm_used,

                # Retrieval confidence
                "top1_score": top1_score,

                # Performance
                "timings": timings_dict,
            }
        )

        logger.info(
            "voice.turn_complete "
            "total_ms=%.2f "
            "generation=%s "
            "llm_used=%s "
            "language=%s",
            collector_total_ms,
            generation_method,
            llm_used,
            language_code,
        )

    # ========================================================
    # Client disconnect
    # ========================================================

    except WebSocketDisconnect:

        logger.info(
            "voice.client_disconnected"
        )

    # ========================================================
    # Unexpected error
    # ========================================================

    except Exception:

        logger.exception(
            "voice.unhandled_error"
        )

        try:

            await ws.send_json(
                {
                    "type": "error",
                    "stage": "internal",
                    "detail": (
                        "unexpected server error"
                    ),
                }
            )

        except Exception:
            pass

    # ========================================================
    # Always close websocket
    # ========================================================

    finally:

        try:
            await ws.close()
        except Exception:
            pass


# ============================================================
# Guardrail error
# ============================================================


def _guardrail_detail(
    exc: GuardrailViolation,
) -> dict:
    """
    Match the HTTP /query guardrail response shape.
    """

    if (
        settings.GUARDRAIL_RESPONSE_DETAIL
        == "detailed"
    ):

        return {
            "check": getattr(
                exc,
                "check",
                "unknown",
            ),

            "detail": str(exc),
        }

    return {
        "detail": (
            "Request blocked by input validation."
        )
    }