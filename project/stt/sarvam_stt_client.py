"""
stt/sarvam_stt_client.py

Sarvam Saaras v3 Realtime STT client.

Browser PCM
    -> FastAPI
    -> Sarvam Realtime WebSocket
    -> partial transcripts
    -> final transcript + detected language
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

import websockets

from project.config.settings import settings
from project.timing.stage_timer import TimingCollector


logger = logging.getLogger("stt.sarvam")


@dataclass
class STTResult:
    transcript: str
    is_final: bool
    language_code: Optional[str] = None


class SarvamSTTError(RuntimeError):
    pass


class SarvamSTTClient:

    def __init__(
        self,
        timing: Optional[TimingCollector] = None,
    ):
        self.timing = timing
        self._first_partial_marked = False

    def session(self):
        return _SarvamSTTSession(self)

    async def _connect(self):

        params = {
            "language_code": settings.SARVAM_STT_LANGUAGE_CODE,
            "model": settings.SARVAM_STT_MODEL,
            "stream_type": "fast",
            "mode": "transcribe",
            "endpointing": "manual",
            "encoding": "linear16",
            "sample_rate": settings.SARVAM_STT_SAMPLE_RATE,
        }

        url = (
            f"{settings.SARVAM_STT_WS_URL}"
            f"?{urlencode(params)}"
        )

        headers = {
            "Api-Subscription-Key": settings.SARVAM_API_KEY,
        }

        logger.info(
            "stt.connect url=%s language=%s model=%s "
            "sample_rate=%s encoding=%s endpointing=manual",
            url,
            settings.SARVAM_STT_LANGUAGE_CODE,
            settings.SARVAM_STT_MODEL,
            settings.SARVAM_STT_SAMPLE_RATE,
            "linear16",
        )

        try:
            ws = await websockets.connect(
                url,
                additional_headers=headers,
                max_size=None,
            )
        except Exception as exc:
            raise SarvamSTTError(
                f"Failed to connect to Sarvam STT: {exc}"
            ) from exc

        logger.info("stt.connected")

        return ws


class _SarvamSTTSession:

    def __init__(self, client: SarvamSTTClient):
        self._client = client
        self._ws = None
        self._started = False
        self._finished = False

    async def __aenter__(self):
        self._ws = await self._client._connect()

        # Manual endpointing requires speech_start.
        await self._ws.send(
            json.dumps({"event": "speech_start"})
        )

        self._started = True

        logger.debug("stt.speech_start_sent")

        return self

    async def __aexit__(self, exc_type, exc, tb):

        if self._ws is not None:

            try:
                if not self._finished:
                    try:
                        await self._ws.send(
                            json.dumps({"event": "end"})
                        )
                    except Exception:
                        pass

                await self._ws.close()

            except Exception:
                logger.exception("stt.close_error")

            finally:
                logger.info("stt.disconnect")

    async def send_audio(self, pcm_bytes: bytes) -> None:

        if self._ws is None:
            raise SarvamSTTError("session not started")

        if self._finished:
            return

        if not pcm_bytes:
            return

        # Sarvam Realtime expects:
        #
        # {
        #     "event": "audio_input",
        #     "audio": "<base64>"
        # }
        message = {
            "event": "audio_input",
            "audio": base64.b64encode(pcm_bytes).decode("utf-8"),
        }

        await self._ws.send(
            json.dumps(message)
        )

    async def finish(self) -> None:

        if self._ws is None:
            raise SarvamSTTError("session not started")

        if self._finished:
            return

        # Manual endpointing:
        #
        # speech_end -> flush -> transcript.final
        await self._ws.send(
            json.dumps({"event": "speech_end"})
        )

        logger.info("stt.speech_end_sent")

        await self._ws.send(
            json.dumps({"event": "flush"})
        )

        logger.info("stt.flush_sent")

        self._finished = True

    async def results(self) -> AsyncIterator[STTResult]:

        if self._ws is None:
            raise SarvamSTTError("session not started")

        try:

            async for raw in self._ws:

                if isinstance(raw, bytes):
                    logger.warning(
                        "stt.unexpected_binary_response bytes=%d",
                        len(raw),
                    )
                    continue

                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):

                    logger.warning(
                        "stt.unparseable_message raw=%r",
                        raw,
                    )

                    continue

                event = msg.get("event")

                logger.debug(
                    "stt.provider_event event=%s",
                    event,
                )

                # -----------------------------------------
                # SESSION
                # -----------------------------------------

                if event == "session.begin":
                    logger.info("stt.session_begin")
                    continue

                if event == "session.end":
                    logger.info(
                        "stt.session_end audio_duration_s=%s",
                        msg.get("audio_duration_s"),
                    )
                    continue

                # -----------------------------------------
                # VAD
                # -----------------------------------------

                if event == "vad.speech_start":
                    logger.debug("stt.vad_speech_start")
                    continue

                if event == "vad.speech_end":
                    logger.debug("stt.vad_speech_end")
                    continue

                # -----------------------------------------
                # PARTIAL TRANSCRIPT
                # -----------------------------------------

                if event == "transcript.partial":

                    transcript = msg.get("text", "")
                    language_code = msg.get("language")

                    if (
                        not self._client._first_partial_marked
                    ):
                        self._client._first_partial_marked = True

                        if self._client.timing:
                            self._client.timing.mark(
                                "stt_first_partial"
                            )

                    logger.info(
                        "stt.partial language=%s text=%r",
                        language_code,
                        transcript,
                    )

                    yield STTResult(
                        transcript=transcript,
                        is_final=False,
                        language_code=language_code,
                    )

                    continue

                # -----------------------------------------
                # FINAL TRANSCRIPT
                # -----------------------------------------

                if event == "transcript.final":

                    transcript = msg.get("text", "")
                    language_code = msg.get("language")

                    if self._client.timing:
                        self._client.timing.mark(
                            "stt_final"
                        )

                    logger.info(
                        "stt.final language=%s "
                        "language_confidence=%s "
                        "transcript=%r",
                        language_code,
                        msg.get("language_confidence"),
                        transcript,
                    )

                    yield STTResult(
                        transcript=transcript,
                        is_final=True,
                        language_code=language_code,
                    )

                    return

                # -----------------------------------------
                # ERROR
                # -----------------------------------------

                if event == "error":

                    code = msg.get("code")
                    message = msg.get(
                        "message",
                        "unknown Sarvam STT error",
                    )

                    fatal = msg.get("is_fatal")

                    logger.error(
                        "stt.provider_error "
                        "code=%s fatal=%s detail=%s",
                        code,
                        fatal,
                        message,
                    )

                    raise SarvamSTTError(
                        f"{message}"
                    )

                # -----------------------------------------
                # UNKNOWN
                # -----------------------------------------

                logger.debug(
                    "stt.unhandled_event payload=%s",
                    msg,
                )

        except websockets.ConnectionClosed as exc:

            logger.warning(
                "stt.connection_closed code=%s reason=%s",
                exc.code,
                exc.reason,
            )

            if not self._finished:
                raise SarvamSTTError(
                    f"Sarvam STT connection closed "
                    f"before final transcript: "
                    f"{exc.code} {exc.reason}"
                ) from exc


async def transcribe_stream(
    audio_frames: AsyncIterator[bytes],
    timing: Optional[TimingCollector] = None,
    stop_event: Optional[asyncio.Event] = None,
) -> STTResult:

    client = SarvamSTTClient(
        timing=timing
    )

    async with client.session() as sess:

        async def _pump_audio():

            frame_count = 0
            total_bytes = 0

            try:

                async for frame in audio_frames:

                    if (
                        stop_event is not None
                        and stop_event.is_set()
                    ):
                        break

                    if not frame:
                        continue

                    frame_count += 1
                    total_bytes += len(frame)

                    if (
                        frame_count <= 5
                        or frame_count % 50 == 0
                    ):
                        logger.info(
                            "stt.audio_frame "
                            "count=%d bytes=%d total_bytes=%d",
                            frame_count,
                            len(frame),
                            total_bytes,
                        )

                    await sess.send_audio(frame)

                logger.info(
                    "stt.audio_finished "
                    "frames=%d total_bytes=%d",
                    frame_count,
                    total_bytes,
                )

                await sess.finish()

            except asyncio.CancelledError:

                logger.debug(
                    "stt.audio_pump_cancelled"
                )

                raise

            except Exception as exc:

                logger.error(
                    "stt.audio_pump_error detail=%s",
                    exc,
                )

                raise

        pump_task = asyncio.create_task(
            _pump_audio()
        )

        final_result: Optional[STTResult] = None

        try:

            async for result in sess.results():

                if result.is_final:

                    final_result = result

                    break

        finally:

            if not pump_task.done():

                pump_task.cancel()

            try:
                await pump_task

            except asyncio.CancelledError:
                pass

            except Exception as exc:

                logger.error(
                    "stt.audio_pump_cleanup_error "
                    "detail=%s",
                    exc,
                )

        if final_result is None:

            raise SarvamSTTError(
                "Sarvam STT closed without "
                "a final transcript"
            )

        return final_result