"""
config/settings.py

Single source of truth for all configuration. Everything here either:
  (a) mirrors a constant that was hardcoded in Cell 1 of Update_version_2.ipynb, or
  (b) is new production config (logging, timing, guardrails, API keys) that did not
      exist in the notebook.

No logic lives here — only configuration loading and defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val is not None else default


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


def _get_list(name: str, default: List[str]) -> List[str]:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return [item.strip() for item in val.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------------
    # Ingestion constants — mirror Cell 1 of the notebook exactly.
    # SAMPLE_SIZE_PER_LANG / QUERY_TYPE_DIST / YEAR_RE-equivalent live here
    # so ingestion/ modules import from one place instead of hardcoding.
    # ------------------------------------------------------------------
    SAMPLE_SIZE_PER_LANG: int = field(
        default_factory=lambda: _get_int("SAMPLE_SIZE_PER_LANG", 3000)
    )
    QUERY_TYPE_DIST: dict = field(
        default_factory=lambda: {
            "DESCRIPTION": 0.529,
            "NUMERIC": 0.263,
            "ENTITY": 0.089,
            "LOCATION": 0.063,
            "PERSON": 0.056,
        }
    )
    DEDUP_THRESHOLD: float = field(
        default_factory=lambda: _get_float("DEDUP_THRESHOLD", 0.9)
    )

    # ------------------------------------------------------------------
    # Storage paths — mirror Cell 1 (UNIFIED_FILE, SQLITE_DB)
    # ------------------------------------------------------------------
    UNIFIED_FILE: str = field(
        default_factory=lambda: os.getenv("UNIFIED_FILE", "unified_corpus.parquet")
    )
    SQLITE_DB: str = field(
        default_factory=lambda: os.getenv("SQLITE_DB", "native_text_lookup.db")
    )

    # ------------------------------------------------------------------
    # Qdrant / embedding models — mirror Cell 1 (QDRANT_COLLECTION,
    # EMBED_MODEL, EMBED_DIM) plus the sparse model added later in the
    # notebook (post-Cell-19 blocks: Qdrant/bm25 via fastembed).
    # ------------------------------------------------------------------
    QDRANT_URL: str = field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    QDRANT_COLLECTION: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "msmarco_english_corpus")
    )
    EMBED_MODEL: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    )
    EMBED_DIM: int = field(default_factory=lambda: _get_int("EMBED_DIM", 384))
    SPARSE_MODEL: str = field(
        default_factory=lambda: os.getenv("SPARSE_MODEL", "Qdrant/bm25")
    )

    INGESTION_BATCH_SIZE: int = field(
    default_factory=lambda: _get_int("INGESTION_BATCH_SIZE", 1000)
)

    EMBED_BATCH_SIZE: int = field(
        default_factory=lambda: _get_int("EMBED_BATCH_SIZE", 32)
    )

    # ------------------------------------------------------------------
    # Retrieval — default mode + hybrid_weighted params (Cell 39)
    # ------------------------------------------------------------------
    RETRIEVAL_MODE: str = field(
        default_factory=lambda: os.getenv("RETRIEVAL_MODE", "hybrid_weighted")
    )
    CONFIDENCE_THRESHOLD: float = field(
        default_factory=lambda: _get_float("CONFIDENCE_THRESHOLD", 0.85)
    )
    HYBRID_DENSE_WEIGHT: float = field(
        default_factory=lambda: _get_float("HYBRID_DENSE_WEIGHT", 0.75)
    )
    HYBRID_SPARSE_WEIGHT: float = field(
        default_factory=lambda: _get_float("HYBRID_SPARSE_WEIGHT", 0.25)
    )
    HYBRID_CANDIDATE_POOL: int = field(
        default_factory=lambda: _get_int("HYBRID_CANDIDATE_POOL", 20)
    )
    TOP_K: int = field(default_factory=lambda: _get_int("TOP_K", 5))

    # ------------------------------------------------------------------
    # LLM fallback (Groq) — from user's draft app.py
    # ------------------------------------------------------------------
    GROQ_API_KEY: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    GROQ_MODEL: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )

    # ------------------------------------------------------------------
    # TTS (ElevenLabs) — RETIRED this phase (see CLAUDE.md §0.1). Kept
    # only so tts/elevenlabs_client.py still imports cleanly for
    # reference; not read by the live path (api/ws_voice.py uses Sarvam).
    # ------------------------------------------------------------------
    ELEVENLABS_API_KEY: str = field(
        default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", "")
    )
    ELEVENLABS_VOICE_ID: str = field(
        default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", "")
    )
    TTS_WORD_LIMIT: int = field(default_factory=lambda: _get_int("TTS_WORD_LIMIT", 200))
    TTS_OUTPUT_DIR: str = field(
        default_factory=lambda: os.getenv("TTS_OUTPUT_DIR", "./tts_out")
    )

    # ------------------------------------------------------------------
    # Sarvam AI — realtime STT + TTS over websocket. NEW this phase
    # (CLAUDE.md §0.1, §8). Replaces ElevenLabs in the live TTS path;
    # STT has no prior-phase equivalent. Both stt/sarvam_stt_client.py
    # and tts/sarvam_tts_client.py read these directly off `settings`.
    # URL/model/speaker values are placeholders — confirm against
    # Sarvam's dashboard/docs before running against the real API
    # (CLAUDE.md §3, §12).
    # ------------------------------------------------------------------
    SARVAM_API_KEY: str = field(default_factory=lambda: os.getenv("SARVAM_API_KEY", ""))

    SARVAM_TRANSLATE_URL: str = os.getenv(
        "SARVAM_TRANSLATE_URL",
        "https://api.sarvam.ai/translate",
    )

    SARVAM_STT_WS_URL: str = field(
        default_factory=lambda: os.getenv(
            "SARVAM_STT_WS_URL", "wss://api.sarvam.ai/speech-to-text-realtime/ws"
        )
    )
    SARVAM_STT_LANGUAGE_CODE: str = field(
        default_factory=lambda: os.getenv("SARVAM_STT_LANGUAGE_CODE", "auto")
    )
    SARVAM_STT_SAMPLE_RATE: int = field(
        default_factory=lambda: _get_int("SARVAM_STT_SAMPLE_RATE", 16000)
    )
    SARVAM_STT_MODEL: str = field(
        default_factory=lambda: os.getenv("SARVAM_STT_MODEL", "saaras:v3-realtime")
    )

    SARVAM_TTS_WS_URL: str = field(
        default_factory=lambda: os.getenv(
            "SARVAM_TTS_WS_URL", "wss://api.sarvam.ai/text-to-speech/ws"
        )
    )
    SARVAM_TTS_MODEL: str = field(
    default_factory=lambda: os.getenv(
        "SARVAM_TTS_MODEL",
        "bulbul:v3"
    )
)
    SARVAM_TTS_SPEAKER: str = field(
        default_factory=lambda: os.getenv("SARVAM_TTS_SPEAKER", "simran")
    )
    SARVAM_TTS_SAMPLE_RATE: int = field(
        default_factory=lambda: _get_int("SARVAM_TTS_SAMPLE_RATE", 16000)
    )
    SARVAM_TTS_LANGUAGE_CODE: str = field(
        default_factory=lambda: os.getenv("SARVAM_TTS_LANGUAGE_CODE", "en-IN")
    )

    # ------------------------------------------------------------------
    # Voice websocket endpoint (api/ws_voice.py) — NEW this phase.
    # ------------------------------------------------------------------
    VOICE_WS_MAX_TURN_SECONDS: int = field(
        default_factory=lambda: _get_int("VOICE_WS_MAX_TURN_SECONDS", 30)
    )

    # ------------------------------------------------------------------
    # Guardrails — NEW, not in the notebook.
    # Checks implemented in guardrails/input_guardrails.py: empty/too-long,
    # non-UTF8/garbage, script/HTML injection, prompt-injection, PII,
    # profanity, blocked topics. Profanity + blocked-topics use wordlist
    # files (one term per line, '#' = comment) rather than env-var lists,
    # so they can be edited without touching config/env at all. Both
    # files are stubbed empty by default (empty wordlist = that check is
    # a no-op) until the user populates them.
    # ------------------------------------------------------------------
    GUARDRAIL_MAX_QUERY_LENGTH: int = field(
        default_factory=lambda: _get_int("GUARDRAIL_MAX_QUERY_LENGTH", 500)
    )
    GUARDRAIL_MIN_QUERY_LENGTH: int = field(
        default_factory=lambda: _get_int("GUARDRAIL_MIN_QUERY_LENGTH", 1)
    )
    GUARDRAIL_PROFANITY_FILE: str = field(
        default_factory=lambda: os.getenv(
            "GUARDRAIL_PROFANITY_FILE", "guardrails/wordlists/profanity.txt"
        )
    )
    GUARDRAIL_BLOCKED_TOPICS_FILE: str = field(
        default_factory=lambda: os.getenv(
            "GUARDRAIL_BLOCKED_TOPICS_FILE", "guardrails/wordlists/blocked_topics.txt"
        )
    )
    # "detailed" -> 404 body names which check failed (e.g. "blocked_topic").
    # "generic"  -> 404 body is a flat, uninformative error message.
    GUARDRAIL_RESPONSE_DETAIL: str = field(
        default_factory=lambda: os.getenv("GUARDRAIL_RESPONSE_DETAIL", "detailed")
    )

    # ------------------------------------------------------------------
    # Logging — NEW
    # ------------------------------------------------------------------
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    LOG_DIR: str = field(default_factory=lambda: os.getenv("LOG_DIR", "./logs"))

    def ensure_dirs(self) -> None:
        """Create output directories that must exist before the app runs."""
        Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.TTS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


settings = Settings()