"""
translation/sarvam_translate_client.py

Sarvam text translation client.

Pipeline:

    User speech
        ↓
    Sarvam STT
        ↓
    Original transcript + language_code
        ↓
    translate_to_english()
        ↓
    English retrieval query
        ↓
    Qdrant / Groq
        ↓
    English answer
        ↓
    translate_answer_to_user_language()
        ↓
    Original user language
        ↓
    Sarvam TTS

Examples:

Hindi:
    transcript:
        "अमेरिका का राष्ट्रपति कौन है?"

    retrieval query:
        "Who is the president of America?"

    final answer:
        "डोनाल्ड ट्रंप अमेरिका के राष्ट्रपति हैं।"

Odia:
    transcript:
        "ମାଣ୍ଟଲରେ ସଂବହନ ସ୍ରୋତ କାହିଁକି ହୁଏ?"

    retrieval query:
        "Why do convection currents occur in the mantle?"

    final answer:
        Odia answer

English:
    Translation API is skipped.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from project.config.settings import settings


logger = logging.getLogger("translation.sarvam")


# ============================================================
# Errors
# ============================================================


class SarvamTranslationError(RuntimeError):
    """Raised when Sarvam Translation fails."""


# ============================================================
# Result
# ============================================================


@dataclass
class TranslationResult:
    translated_text: str
    source_language_code: str
    target_language_code: str
    translation_ms: float


# ============================================================
# Generic translation
# ============================================================


async def translate_text(
    text: str,
    source_language_code: str,
    target_language_code: str,
) -> TranslationResult:
    """
    Generic Sarvam translation.

    Parameters
    ----------
    text:
        Text that should be translated.

    source_language_code:
        Source language, for example:
            en-IN
            hi-IN
            od-IN
            ta-IN

    target_language_code:
        Target language, for example:
            en-IN
            hi-IN
            od-IN
            ta-IN

    Returns
    -------
    TranslationResult
    """

    # --------------------------------------------------------
    # Clean input
    # --------------------------------------------------------

    text = (text or "").strip()

    if not text:
        raise SarvamTranslationError(
            "Cannot translate empty text"
        )

    source_language_code = (
        source_language_code or "en-IN"
    )

    target_language_code = (
        target_language_code or "en-IN"
    )

    # --------------------------------------------------------
    # Same-language optimization
    #
    # English -> English
    # Hindi -> Hindi
    # etc.
    #
    # No API request required.
    # --------------------------------------------------------

    if (
        source_language_code
        == target_language_code
    ):

        logger.info(
            "translation.skipped "
            "source=%s "
            "target=%s "
            "reason=same_language",
            source_language_code,
            target_language_code,
        )

        return TranslationResult(
            translated_text=text,
            source_language_code=source_language_code,
            target_language_code=target_language_code,
            translation_ms=0.0,
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    url = settings.SARVAM_TRANSLATE_URL

    api_key = settings.SARVAM_API_KEY

    model = getattr(
        settings,
        "SARVAM_TRANSLATE_MODEL",
        "sarvam-translate:v1",
    )

    if not api_key:
        raise SarvamTranslationError(
            "SARVAM_API_KEY is not configured"
        )

    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "input": text,
        "source_language_code":
            source_language_code,
        "target_language_code":
            target_language_code,
        "model": model,
    }

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logger.info(
        "translation.start "
        "source=%s "
        "target=%s "
        "model=%s "
        "chars=%d",
        source_language_code,
        target_language_code,
        model,
        len(text),
    )

    logger.debug(
        "translation.input "
        "source=%s "
        "target=%s "
        "text=%r",
        source_language_code,
        target_language_code,
        text,
    )

    start = time.perf_counter()

    # --------------------------------------------------------
    # API request
    # --------------------------------------------------------

    try:

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=30.0,
                write=10.0,
                pool=10.0,
            )
        ) as client:

            response = await client.post(
                url,
                headers=headers,
                json=payload,
            )

        translation_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        # ----------------------------------------------------
        # Provider error
        # ----------------------------------------------------

        if response.status_code != 200:

            logger.error(
                "translation.provider_error "
                "status=%d "
                "source=%s "
                "target=%s "
                "duration_ms=%.2f "
                "body=%s",
                response.status_code,
                source_language_code,
                target_language_code,
                translation_ms,
                response.text[:1000],
            )

            raise SarvamTranslationError(
                f"Sarvam translation failed "
                f"(HTTP {response.status_code}): "
                f"{response.text[:500]}"
            )

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        try:

            data = response.json()

        except ValueError as exc:

            logger.error(
                "translation.invalid_json "
                "source=%s "
                "target=%s "
                "duration_ms=%.2f "
                "body=%s",
                source_language_code,
                target_language_code,
                translation_ms,
                response.text[:500],
            )

            raise SarvamTranslationError(
                "Sarvam returned invalid JSON"
            ) from exc

        # ----------------------------------------------------
        # Extract translated text
        # ----------------------------------------------------

        translated_text = (
            data.get("translated_text")
            or ""
        ).strip()

        if not translated_text:

            logger.error(
                "translation.empty_response "
                "source=%s "
                "target=%s "
                "duration_ms=%.2f "
                "response=%s",
                source_language_code,
                target_language_code,
                translation_ms,
                data,
            )

            raise SarvamTranslationError(
                "Sarvam returned empty translated_text"
            )

        # ----------------------------------------------------
        # Success logging
        # ----------------------------------------------------

        logger.info(
            "translation.complete "
            "source=%s "
            "target=%s "
            "duration_ms=%.2f "
            "input_chars=%d "
            "output_chars=%d",
            source_language_code,
            target_language_code,
            translation_ms,
            len(text),
            len(translated_text),
        )

        logger.info(
            "translation.result "
            "source=%s "
            "target=%s "
            "source_text=%r "
            "translated_text=%r",
            source_language_code,
            target_language_code,
            text,
            translated_text,
        )

        return TranslationResult(
            translated_text=translated_text,
            source_language_code=source_language_code,
            target_language_code=target_language_code,
            translation_ms=round(
                translation_ms,
                2,
            ),
        )

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    except httpx.TimeoutException as exc:

        translation_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        logger.error(
            "translation.timeout "
            "source=%s "
            "target=%s "
            "duration_ms=%.2f",
            source_language_code,
            target_language_code,
            translation_ms,
        )

        raise SarvamTranslationError(
            "Sarvam translation request timed out"
        ) from exc

    # --------------------------------------------------------
    # HTTP/network error
    # --------------------------------------------------------

    except httpx.HTTPError as exc:

        translation_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        logger.error(
            "translation.http_error "
            "source=%s "
            "target=%s "
            "duration_ms=%.2f "
            "detail=%s",
            source_language_code,
            target_language_code,
            translation_ms,
            exc,
        )

        raise SarvamTranslationError(
            f"Sarvam translation HTTP error: {exc}"
        ) from exc


# ============================================================
# User transcript -> English
# ============================================================


async def translate_to_english(
    text: str,
    source_language_code: str,
) -> TranslationResult:
    """
    Translate the user's final STT transcript to English.

    This translation is used ONLY for retrieval.

    Example:

        Hindi:
            "अमेरिका का राष्ट्रपति कौन है?"

        English:
            "Who is the president of America?"
    """

    source_language_code = (
        source_language_code or "en-IN"
    )

    return await translate_text(
        text=text,
        source_language_code=source_language_code,
        target_language_code="en-IN",
    )


# ============================================================
# English answer -> User language
# ============================================================


async def translate_answer_to_user_language(
    text: str,
    target_language_code: str,
) -> TranslationResult:
    """
    Translate the generated English answer into
    the language detected from the user's speech.

    Example:

        English answer:
            "The President of the United States is ..."

        target_language_code:
            "hi-IN"

        result:
            Hindi answer
    """

    target_language_code = (
        target_language_code or "en-IN"
    )

    return await translate_text(
        text=text,
        source_language_code="en-IN",
        target_language_code=target_language_code,
    )