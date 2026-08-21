"""
guardrails/__init__.py

Public surface for the input-guardrails module (CLAUDE.md phase 5).
"""
from project.guardrails.input_guardrails import (
    GuardrailResult,
    GuardrailViolation,
    run_guardrails,
    run_guardrails_strict,
    build_error_response,
    check_empty,
    check_too_long,
    check_non_utf8_garbage,
    check_script_html_injection,
    check_prompt_injection,
    check_pii,
    check_profanity,
    check_blocked_topics,
)

__all__ = [
    "GuardrailResult",
    "GuardrailViolation",
    "run_guardrails",
    "run_guardrails_strict",
    "build_error_response",
    "check_empty",
    "check_too_long",
    "check_non_utf8_garbage",
    "check_script_html_injection",
    "check_prompt_injection",
    "check_pii",
    "check_profanity",
    "check_blocked_topics",
]
