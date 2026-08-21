"""
tts/sarvam_tts_client.py

Sarvam Bulbul v3 realtime WebSocket TTS.

Flow:

    connect
       ↓
    configure
       ↓
    send text
       ↓
    flush
       ↓
    receive audio chunks
       ↓
    receive event.final
       ↓
    return / close websocket

Important:
Sarvam Bulbul v3 WebSocket completion is represented by:

{
    "type": "event",
    "data": {
        "event_type": "final"
    }
}

Do NOT wait for "end_of_audio".
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

import websockets

from project.config.settings import settings
from project.timing.stage_timer import TimingCollector


logger = logging.getLogger("tts.sarvam")


class SarvamTTSError(RuntimeError):
    pass


async def synthesize_stream(
    text: str,
    language_code: Optional[str] = None,
    timing: Optional[TimingCollector] = None,
) -> AsyncIterator[bytes]:
    """
    Stream text through Sarvam Bulbul v3 and yield audio chunks.

    The websocket is closed immediately after Sarvam sends
    the final completion event.
    """

    if not text or not text.strip():
        raise SarvamTTSError(
            "Cannot synthesize empty text"
        )

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    language = (
        language_code
        or settings.SARVAM_TTS_LANGUAGE_CODE
        or "en-IN"
    )

    # --------------------------------------------------------
    # Query parameters
    # --------------------------------------------------------

    params = {
        "model": settings.SARVAM_TTS_MODEL,
        "send_completion_event": "true",
    }

    url = (
        f"{settings.SARVAM_TTS_WS_URL}"
        f"?{urlencode(params)}"
    )

    headers = {
        "Api-Subscription-Key": settings.SARVAM_API_KEY,
    }

    logger.info(
        "tts.connect "
        "url=%s "
        "model=%s "
        "speaker=%s "
        "language=%s "
        "chars=%d",
        url,
        settings.SARVAM_TTS_MODEL,
        settings.SARVAM_TTS_SPEAKER,
        language,
        len(text),
    )

    connect_start = time.perf_counter()

    first_chunk_marked = False
    chunk_count = 0
    total_audio_bytes = 0

    try:

        # ----------------------------------------------------
        # WebSocket connection
        # ----------------------------------------------------

        async with websockets.connect(
            url,
            additional_headers=headers,
            max_size=None,
        ) as ws:

            connect_ms = (
                time.perf_counter()
                - connect_start
            ) * 1000

            logger.info(
                "tts.connected "
                "connect_ms=%.2f",
                connect_ms,
            )

            # =================================================
            # CONFIG
            # =================================================

            config_message = {
                "type": "config",
                "data": {
                    "speaker": (
                        settings.SARVAM_TTS_SPEAKER
                    ),
                    "language_code": language,

                    # Bulbul v3 default is 24000.
                    # Keep this aligned with settings.
                    "output_audio_codec": "linear16",

                    "sample_rate": (
                        settings.SARVAM_TTS_SAMPLE_RATE
                    ),
                },
            }

            await ws.send(
                json.dumps(
                    config_message
                )
            )

            logger.debug(
                "tts.config_sent"
            )

            # =================================================
            # TEXT
            # =================================================

            text_message = {
                "type": "text",
                "data": {
                    "text": text,
                },
            }

            await ws.send(
                json.dumps(
                    text_message
                )
            )

            logger.debug(
                "tts.text_sent chars=%d",
                len(text),
            )

            # =================================================
            # FLUSH
            # =================================================

            flush_message = {
                "type": "flush",
            }

            await ws.send(
                json.dumps(
                    flush_message
                )
            )

            logger.info(
                "tts.flush_sent"
            )

            # =================================================
            # RECEIVE LOOP
            # =================================================

            async for raw in ws:

                # ------------------------------------------------
                # Binary message
                # ------------------------------------------------

                if isinstance(
                    raw,
                    bytes,
                ):

                    chunk = raw

                    if not chunk:
                        continue

                    chunk_count += 1
                    total_audio_bytes += len(
                        chunk
                    )

                    if not first_chunk_marked:

                        first_chunk_marked = True

                        if timing:
                            timing.mark(
                                "tts_first_chunk"
                            )

                        logger.info(
                            "tts.first_chunk "
                            "bytes=%d "
                            "chunk=%d",
                            len(chunk),
                            chunk_count,
                        )

                    logger.debug(
                        "tts.audio_chunk "
                        "chunk=%d "
                        "bytes=%d "
                        "total_bytes=%d",
                        chunk_count,
                        len(chunk),
                        total_audio_bytes,
                    )

                    yield chunk

                    continue

                # ------------------------------------------------
                # JSON message
                # ------------------------------------------------

                try:

                    msg = json.loads(
                        raw
                    )

                except (
                    json.JSONDecodeError,
                    TypeError,
                ):

                    logger.warning(
                        "tts.unparseable_message "
                        "type=%s",
                        type(raw).__name__,
                    )

                    continue

                message_type = msg.get(
                    "type"
                )

                # =================================================
                # AUDIO
                # =================================================

                if message_type == "audio":

                    data = msg.get(
                        "data",
                        {},
                    )

                    b64_audio = data.get(
                        "audio"
                    )

                    if not b64_audio:
                        logger.warning(
                            "tts.audio_message_without_audio"
                        )
                        continue

                    try:

                        chunk = base64.b64decode(
                            b64_audio
                        )

                    except Exception as exc:

                        raise SarvamTTSError(
                            f"Invalid audio payload: {exc}"
                        ) from exc

                    if not chunk:
                        continue

                    chunk_count += 1
                    total_audio_bytes += len(
                        chunk
                    )

                    if not first_chunk_marked:

                        first_chunk_marked = True

                        if timing:
                            timing.mark(
                                "tts_first_chunk"
                            )

                        logger.info(
                            "tts.first_chunk "
                            "bytes=%d "
                            "chunk=%d",
                            len(chunk),
                            chunk_count,
                        )

                    logger.debug(
                        "tts.audio_chunk "
                        "chunk=%d "
                        "bytes=%d "
                        "total_bytes=%d",
                        chunk_count,
                        len(chunk),
                        total_audio_bytes,
                    )

                    yield chunk

                    continue

                # =================================================
                # COMPLETION EVENT
                # =================================================

                if message_type == "event":

                    data = msg.get(
                        "data",
                        {},
                    )

                    event_type = data.get(
                        "event_type"
                    )

                    logger.info(
                        "tts.event "
                        "event_type=%s",
                        event_type,
                    )

                    if event_type == "final":

                        if timing:
                            timing.mark(
                                "tts_last_chunk"
                            )

                        logger.info(
                            "tts.complete "
                            "chunks=%d "
                            "total_audio_bytes=%d",
                            chunk_count,
                            total_audio_bytes,
                        )

                        # IMPORTANT:
                        # Return immediately.
                        # Do not wait for another message.
                        return

                    continue

                # =================================================
                # ERROR
                # =================================================

                if message_type == "error":

                    data = msg.get(
                        "data",
                        {},
                    )

                    detail = (
                        data.get(
                            "message"
                        )
                        or msg.get(
                            "message"
                        )
                        or "Unknown Sarvam TTS error"
                    )

                    code = (
                        data.get(
                            "code"
                        )
                        or msg.get(
                            "code"
                        )
                    )

                    logger.error(
                        "tts.provider_error "
                        "code=%s "
                        "detail=%s",
                        code,
                        detail,
                    )

                    raise SarvamTTSError(
                        detail
                    )

                # =================================================
                # UNKNOWN MESSAGE
                # =================================================

                logger.debug(
                    "tts.unknown_message "
                    "type=%s",
                    message_type,
                )

    # ------------------------------------------------------------
    # WebSocket closed
    # ------------------------------------------------------------

    except websockets.ConnectionClosed as exc:

        logger.warning(
            "tts.connection_closed "
            "code=%s "
            "reason=%s",
            exc.code,
            exc.reason,
        )

        # If final event was never received, this is abnormal.
        if timing and not timing.has_mark(
            "tts_last_chunk"
        ):

            timing.mark(
                "tts_last_chunk"
            )

        if not first_chunk_marked:

            raise SarvamTTSError(
                "Sarvam TTS websocket closed "
                "before receiving audio"
            ) from exc

        # Audio was received, but completion was
        # not received.
        raise SarvamTTSError(
            "Sarvam TTS websocket closed "
            "without a final completion event"
        ) from exc

    except SarvamTTSError:
        raise

    except Exception as exc:

        logger.exception(
            "tts.unexpected_error"
        )

        raise SarvamTTSError(
            str(exc)
        ) from exc