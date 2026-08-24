"""
evaluation_adapter/generator.py

Satisfies rag-local-eval-loop's generator interface (TARGET_INTERFACE.md):

    generate_answer(query: str, results: list) -> answer object
        answer.text: str
        answer.grounded: bool
        answer.generation_ms: float
        answer.model: str

`results` items only have to expose `.text` / `.source` per the contract,
but the eval suite's own runtime object (eval/pipeline.py's `_Context`)
also carries `.score` -- the cosine similarity from its own throwaway
FAISS index (METRIC_INNER_PRODUCT over the vectors this project's
evaluation_adapter.embedder returns). This adapter reads that `.score`
defensively (getattr(..., None)) so it still works if some future caller
passes plain `.text`/`.source`-only objects.

Reuses HHGoa's existing production pieces rather than re-implementing
generation:
  - project.config.settings.settings.CONFIDENCE_THRESHOLD -- same
    threshold project/api/text.py's confidence gate uses (default 0.85).
  - project.llm_fallback.groq_client.call_groq -- the same Groq call
    production code makes on the low-confidence path. call_groq expects
    Qdrant-ScoredPoint-shaped objects (`.payload.get("text_en")`); since
    the eval suite hands us plain `.text` objects instead, _AsPayload
    below is a minimal shim translating one attribute shape to the
    other. No Groq/prompt logic is duplicated.

Grounded signal (TARGET_INTERFACE.md step 8 / eval/checks/reliability.py):
Production's confidence gate (project/api/text.py, project/api/app.py)
always produces *some* output -- on the high-confidence branch it echoes
retrieved text, on the low-confidence branch Groq composes an answer from
context, but Groq's system prompt explicitly allows it to reply "the
context does not contain the answer" in that composed text. Production
never turns that into a boolean anywhere; there's no `grounded` field in
today's HHGoa. Rather than hardcoding `grounded = True` (explicitly
disallowed by TARGET_INTERFACE.md), this adapter derives it from two
things production *does* give us:

  1. High-confidence branch (top1 score >= CONFIDENCE_THRESHOLD): the
     retrieval system found a genuinely close match, so grounded=True
     without needing to inspect any generated text.
  2. Low-confidence / no-results branch: Groq is called, and its answer
     text is scanned for the same kind of refusal language its own
     system prompt invites ("the context does not contain...", "I don't
     know", etc. -- see _REFUSAL_PATTERNS). A match means the system
     itself decided it could not answer -> grounded=False.

This is a heuristic, not a certainty -- documented here rather than
silently baked in, per TARGET_INTERFACE.md's warning that a generator
reporting grounded=True unconditionally can never be caught fabricating.
If HHGoa later adds an explicit grounded/refused signal to its own
generation path, this heuristic should be replaced by reading that
signal directly instead.

Does not route through STT/TTS/WebSocket/frontend -- calls
project.llm_fallback.groq_client directly, per TARGET_INTERFACE.md /
the integration brief's step 7.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from project.config.settings import settings
from project.llm_fallback.groq_client import call_groq

_REFUSAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"context does(?:n't| not) contain",
        r"context doesn't (?:have|cover)",
        r"no (?:relevant )?information (?:is )?(?:available|found) (?:in|about)",
        r"(?:i|we) don'?t (?:know|have (?:enough|sufficient) information)",
        r"cannot (?:answer|determine|find)",
        r"can'?t (?:answer|determine|find) (?:this|that)",
        r"not (?:mentioned|covered|found|present) in the (?:provided )?context",
        r"unable to (?:answer|find|determine)",
        r"does not (?:provide|contain) (?:enough|sufficient|any) information",
    ]
]


def _looks_like_refusal(text: str) -> bool:
    if not text:
        return True
    return any(p.search(text) for p in _REFUSAL_PATTERNS)


@dataclass
class GeneratedAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


class _AsPayload:
    """Shim adapting an eval-suite context object (`.text`/`.source`) to
    the `.payload.get("text_en")` shape project.llm_fallback.groq_client
    expects from a Qdrant ScoredPoint. No behavior change to call_groq
    itself -- just translating the attribute shape."""

    def __init__(self, text: str):
        self.payload = {"text_en": text or ""}


def _top1_score(results) -> float | None:
    if not results:
        return None
    return getattr(results[0], "score", None)


def generate_answer(query: str, results: list) -> GeneratedAnswer:
    top1_score = _top1_score(results)

    if results and top1_score is not None and top1_score >= settings.CONFIDENCE_THRESHOLD:
        # High confidence: mirrors project/api/text.py's confidence-gate
        # HIGH_CONFIDENCE branch -- use the retrieved text directly,
        # skip the LLM call entirely.
        t0 = time.perf_counter()
        answer_text = results[0].text
        generation_ms = (time.perf_counter() - t0) * 1000.0
        return GeneratedAnswer(
            text=answer_text,
            grounded=True,
            generation_ms=generation_ms,
            model="hhgoa-retrieval-direct",
        )

    # Low confidence (or no score available, or empty results): mirrors
    # the LOW_CONFIDENCE branch -- Groq composes an answer from context.
    shimmed_results = [_AsPayload(getattr(r, "text", "")) for r in results]
    t0 = time.perf_counter()
    groq_result = call_groq(query, shimmed_results)
    generation_ms = (time.perf_counter() - t0) * 1000.0

    answer_text = groq_result["answer"] or ""
    grounded = bool(results) and not _looks_like_refusal(answer_text)

    return GeneratedAnswer(
        text=answer_text,
        grounded=grounded,
        generation_ms=generation_ms,
        model=settings.GROQ_MODEL,
    )