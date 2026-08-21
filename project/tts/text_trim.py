"""
tts/text_trim.py

Sentence-boundary trim used when the confidence gate is skipped
(top1_score >= settings.CONFIDENCE_THRESHOLD, CLAUDE.md decision #3):
the raw retrieved text is trimmed to a max word count without cutting
a sentence in half, then sent straight to TTS -- no LLM call needed.

NEW module -- not from the notebook. Ported from the user's own draft
app.py, wired against this project's config/logging.
"""

import re
import time

from project.config.settings import settings
from project.logging_system.logger import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def trim_to_word_limit(text: str, word_limit: int = None) -> dict:
    """
    Trim `text` to at most `word_limit` words, stopping at the last
    complete sentence boundary that fits (never cuts a sentence
    mid-way). Falls back to a hard word-count truncation only if a
    single sentence alone exceeds the limit.

    Returns:
        {"text": <str>, "trim_ms": <float>}
    """
    word_limit = word_limit or settings.TTS_WORD_LIMIT

    t0 = time.perf_counter()

    words = text.split()
    if len(words) <= word_limit:
        trimmed = text.strip()
    else:
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        kept = []
        word_count = 0
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if word_count + sentence_words > word_limit and kept:
                break
            kept.append(sentence)
            word_count += sentence_words
            if word_count >= word_limit:
                break

        if kept:
            trimmed = " ".join(kept)
        else:
            # A single sentence alone exceeds the limit -- hard-truncate as last resort.
            trimmed = " ".join(words[:word_limit])

    t1 = time.perf_counter()
    trim_ms = (t1 - t0) * 1000

    logger.debug("trim_to_word_limit: word_limit=%s trim_ms=%.2f", word_limit, trim_ms)

    return {
        "text": trimmed,
        "trim_ms": trim_ms,
    }