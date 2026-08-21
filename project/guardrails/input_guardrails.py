"""
guardrails/input_guardrails.py

Input guardrails for the retrieval pipeline.

Runs on the raw text query before retrieval / embedding.

Checks:
    1. Empty query
    2. Too-long query
    3. Non-UTF8 / garbage input
    4. Script / HTML injection
    5. Prompt injection
    6. PII
    7. Profanity
    8. Blocked topics

On failure, callers can return the configured guardrail error response.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from project.config.settings import settings

GUARDRAIL_MAX_QUERY_LENGTH = settings.GUARDRAIL_MAX_QUERY_LENGTH
GUARDRAIL_PROFANITY_FILE = settings.GUARDRAIL_PROFANITY_FILE
GUARDRAIL_BLOCKED_TOPICS_FILE = settings.GUARDRAIL_BLOCKED_TOPICS_FILE
GUARDRAIL_RESPONSE_DETAIL = settings.GUARDRAIL_RESPONSE_DETAIL


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------
# GUARDRAIL_MAX_QUERY_LENGTH is intentionally optional in settings.py.
# If it is not configured, use this conservative fallback.
_DEFAULT_MAX_QUERY_LENGTH = 500


# ---------------------------------------------------------------------------
# Result / exception types
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    passed: bool
    failed_check: Optional[str] = None
    message: Optional[str] = None


class GuardrailViolation(Exception):
    """Raised by run_guardrails_strict() when a guardrail check fails."""

    def __init__(self, check_name: str, message: str):
        self.check_name = check_name
        self.message = message
        super().__init__(f"[{check_name}] {message}")


# ---------------------------------------------------------------------------
# Wordlist loading
# ---------------------------------------------------------------------------

_wordlist_cache: dict[str, List[str]] = {}


def _load_wordlist(path: str) -> List[str]:
    """
    Load a wordlist file.

    Format:
        - One entry per line
        - Blank lines are ignored
        - Lines beginning with '#' are comments

    Missing or empty files result in an empty list, making the
    corresponding check a no-op.
    """
    if path in _wordlist_cache:
        return _wordlist_cache[path]

    entries: List[str] = []

    p = Path(path)

    if p.exists():
        for line in p.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            line = line.strip()

            if line and not line.startswith("#"):
                entries.append(line.lower())

    _wordlist_cache[path] = entries

    return entries


def _contains_any_substring(
    text_lower: str,
    terms: List[str],
) -> Optional[str]:
    """Return the first matching substring, if any."""
    for term in terms:
        if term in text_lower:
            return term

    return None


def _contains_any_word(
    text_lower: str,
    terms: List[str],
) -> Optional[str]:
    """Return the first matching whole word, if any."""
    for term in terms:
        if re.search(
            rf"\b{re.escape(term)}\b",
            text_lower,
        ):
            return term

    return None


# ---------------------------------------------------------------------------
# Prompt-injection patterns
# ---------------------------------------------------------------------------

_PROMPT_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"ignore (all|any)?\s*(previous|prior|above|earlier)\s*(instructions?|prompts?|rules?)",
        r"disregard (all|any)?\s*(previous|prior|above|earlier)",
        r"forget (all|everything|what)\s*(you were told|your instructions)?",
        r"you are now (a|an)?\s*\w+",
        r"act as (if )?you (are|were)",
        r"pretend (that )?you are",
        r"reveal (your|the) (system )?(prompt|instructions)",
        r"(show|print|output) (me )?(your|the) (system )?prompt",
        r"new instructions?\s*:",
        r"override (your|the|any) (previous )?instructions?",
        r"bypass (your|the) (safety|guardrails?|restrictions?|filters?)",
        r"\bjailbreak\b",
        r"\bDAN\b.{0,20}(mode|prompt)",
        r"do anything now",
        r"developer mode",
        r"you (have no|don't have) (restrictions|rules|guidelines)",
    ]
]


# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

_PII_PATTERNS = {
    "email": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    ),
    "phone": re.compile(
        r"(\+?\d{1,3}[-.\s]?)?"
        r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "ssn": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    "credit_card": re.compile(
        r"\b(?:\d[ -]?){13,16}\b"
    ),
}


# ---------------------------------------------------------------------------
# Script / HTML injection patterns
# ---------------------------------------------------------------------------

_SCRIPT_HTML_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"<\s*script\b",
        r"<\s*iframe\b",
        r"javascript\s*:",
        r"on\w+\s*=\s*['\"]",
        r"<\s*img[^>]+onerror",
        r"<\s*/?\s*(svg|object|embed)\b",
    ]
]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_empty(query: str) -> GuardrailResult:
    """Reject empty or whitespace-only queries."""
    if query is None or query.strip() == "":
        return GuardrailResult(
            False,
            "empty_query",
            "Query is empty.",
        )

    return GuardrailResult(True)


def check_too_long(
    query: str,
    max_length: Optional[int] = None,
) -> GuardrailResult:
    """
    Reject queries exceeding the configured maximum length.

    If GUARDRAIL_MAX_QUERY_LENGTH is not configured, use the local
    fallback value.
    """
    limit = (
        max_length
        if max_length is not None
        else (
            GUARDRAIL_MAX_QUERY_LENGTH
            or _DEFAULT_MAX_QUERY_LENGTH
        )
    )

    if len(query) > limit:
        return GuardrailResult(
            False,
            "too_long_query",
            f"Query exceeds max length of {limit} characters.",
        )

    return GuardrailResult(True)


def check_non_utf8_garbage(query: str) -> GuardrailResult:
    """
    Reject invalid UTF-8 and excessive control-character input.
    """
    try:
        query.encode("utf-8").decode("utf-8")
    except UnicodeError:
        return GuardrailResult(
            False,
            "non_utf8_input",
            "Query contains invalid UTF-8.",
        )

    if not query:
        return GuardrailResult(True)

    control_chars = sum(
        1
        for ch in query
        if unicodedata.category(ch) == "Cc"
        and ch not in ("\n", "\t")
    )

    if control_chars / max(len(query), 1) > 0.1:
        return GuardrailResult(
            False,
            "garbage_input",
            "Query contains an excessive ratio of control characters.",
        )

    return GuardrailResult(True)


def check_script_html_injection(
    query: str,
) -> GuardrailResult:
    """Reject obvious script / HTML injection patterns."""
    for pattern in _SCRIPT_HTML_PATTERNS:
        if pattern.search(query):
            return GuardrailResult(
                False,
                "script_html_injection",
                "Query contains script/HTML injection patterns.",
            )

    return GuardrailResult(True)


def check_prompt_injection(
    query: str,
) -> GuardrailResult:
    """Reject common prompt-injection patterns."""
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(query):
            return GuardrailResult(
                False,
                "prompt_injection",
                "Query matches a known prompt-injection pattern.",
            )

    return GuardrailResult(True)


def check_pii(query: str) -> GuardrailResult:
    """Reject queries containing supported PII patterns."""
    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(query):
            return GuardrailResult(
                False,
                "pii_detected",
                f"Query appears to contain {pii_type}.",
            )

    return GuardrailResult(True)


def check_profanity(query: str) -> GuardrailResult:
    """
    Check the configured profanity wordlist.

    An empty or missing wordlist is intentionally a no-op.
    """
    terms = _load_wordlist(GUARDRAIL_PROFANITY_FILE)

    if not terms:
        return GuardrailResult(True)

    hit = _contains_any_word(
        query.lower(),
        terms,
    )

    if hit:
        return GuardrailResult(
            False,
            "profanity",
            "Query contains profanity.",
        )

    return GuardrailResult(True)


def check_blocked_topics(
    query: str,
) -> GuardrailResult:
    """
    Check the configured blocked-topic wordlist.

    An empty or missing wordlist is intentionally a no-op.
    """
    terms = _load_wordlist(
        GUARDRAIL_BLOCKED_TOPICS_FILE
    )

    if not terms:
        return GuardrailResult(True)

    hit = _contains_any_substring(
        query.lower(),
        terms,
    )

    if hit:
        return GuardrailResult(
            False,
            "blocked_topic",
            "Query matches a blocked topic.",
        )

    return GuardrailResult(True)


# ---------------------------------------------------------------------------
# Guardrail execution order
# ---------------------------------------------------------------------------

_CHECK_ORDER = [
    check_empty,
    check_too_long,
    check_non_utf8_garbage,
    check_script_html_injection,
    check_prompt_injection,
    check_pii,
    check_profanity,
    check_blocked_topics,
]


def run_guardrails(
    query: str,
) -> GuardrailResult:
    """
    Run all guardrail checks in order.

    Stops at the first failed check.
    """
    for check_fn in _CHECK_ORDER:
        result = check_fn(query)

        if not result.passed:
            return result

    return GuardrailResult(True)


def run_guardrails_strict(
    query: str,
) -> None:
    """
    Run guardrails and raise GuardrailViolation on failure.
    """
    result = run_guardrails(query)

    if not result.passed:
        raise GuardrailViolation(
            result.failed_check,
            result.message,
        )


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------

def build_error_response(
    result_or_violation,
) -> dict:
    """
    Build the JSON response for a guardrail failure.

    generic:
        {
            "error": "Request blocked by input validation."
        }

    detailed:
        {
            "error": "guardrail_violation",
            "check": "<check name>",
            "detail": "<message>"
        }
    """
    if isinstance(
        result_or_violation,
        GuardrailViolation,
    ):
        check_name = result_or_violation.check_name
        message = result_or_violation.message
    else:
        check_name = result_or_violation.failed_check
        message = result_or_violation.message

    if GUARDRAIL_RESPONSE_DETAIL == "detailed":
        return {
            "error": "guardrail_violation",
            "check": check_name,
            "detail": message,
        }

    return {
        "error": "Request blocked by input validation."
    }