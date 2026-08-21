"""
tts/elevenlabs_client.py

ElevenLabs text-to-speech. Takes the final answer text (either the
Groq-composed answer or the sentence-trimmed retrieved text -- see
llm_fallback/groq_client.py and tts/text_trim.py) and saves it as an
mp3 under settings.TTS_OUTPUT_DIR.

NEW module -- not from the notebook. Ported from the user's own draft
app.py, wired against this project's config/logging.
"""

import time
import uuid
from pathlib import Path
import inspect

from elevenlabs.client import ElevenLabs

from project.config.settings import settings
from project.logging_system.logger import get_logger

logger = get_logger(__name__)

_elevenlabs_client = None


def get_elevenlabs_client() -> ElevenLabs:
    """Lazily create and cache the ElevenLabs client."""
    global _elevenlabs_client
    if _elevenlabs_client is None:
        _elevenlabs_client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
    return _elevenlabs_client


def call_elevenlabs_tts(text: str, voice_id: str = "EXAVITQu4vr4xnSDxMaL", output_dir: str = None) -> dict:
    """
    Generate speech audio for `text` and save it as an mp3.

    Returns:
        {"audio_path": <str>, "tts_ms": <float>}
    """
    voice_id = voice_id or settings.ELEVENLABS_VOICE_ID
    output_dir = output_dir or settings.TTS_OUTPUT_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.mp3"
    output_path = Path(output_dir) / filename

    client = get_elevenlabs_client()

    t0 = time.perf_counter()
    try:
        print("VOICE_ID:", repr(settings.ELEVENLABS_VOICE_ID))
        logger.info("VOICE_ID: %r", settings.ELEVENLABS_VOICE_ID)
        logger.info("convert signature: %s", inspect.signature(client.text_to_speech.convert))
        audio = client.text_to_speech.convert(
            voice_id="EXAVITQu4vr4xnSDxMaL",
            text=text,
            model_id="eleven_multilingual_v2",
        )
        with open(output_path, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
    except Exception:
        logger.exception("ElevenLabs TTS call failed")
        raise
    t1 = time.perf_counter()

    tts_ms = (t1 - t0) * 1000
    logger.info("TTS generated: %s (tts_ms=%.2f)", output_path, tts_ms)

    return {
        "audio_path": str(output_path),
        "tts_ms": tts_ms,
    }