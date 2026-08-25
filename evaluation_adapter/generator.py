
"""
evaluation_adapter/generator.py

Evaluation adapter for HHGoa.

Policy:
1. Strong retrieval -> return retrieved passage directly.
2. Medium retrieval -> allow Groq to answer, but only if the
   retrieved evidence is reasonably strong.
3. Weak retrieval -> abstain.

The goal is to reduce false confidence without causing the
system to refuse every answerable query.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from project.config.settings import settings
from project.llm_fallback.groq_client import call_groq


# ------------------------------------------------------------------
# Evaluation thresholds
# ------------------------------------------------------------------

DIRECT_THRESHOLD = 0.90
GROQ_THRESHOLD = 0.75
AMBIGUOUS_MARGIN = 0.015
AMBIGUOUS_SCORE = 0.90


# ------------------------------------------------------------------
# Refusal detection
# ------------------------------------------------------------------

_REFUSAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"context does(?:n't| not) contain",
        r"context doesn't (?:have|cover)",
        r"no (?:relevant )?information (?:is )?(?:available|found)",
        r"(?:i|we) don'?t (?:know|have)",
        r"cannot (?:answer|determine|find)",
        r"can'?t (?:answer|determine|find)",
        r"not (?:mentioned|covered|found|present) in",
        r"unable to (?:answer|find|determine)",
        r"does not (?:provide|contain)",
        r"not enough information",
        r"insufficient information",
    ]
]


def _looks_like_refusal(text: str) -> bool:
    if not text:
        return True

    return any(pattern.search(text) for pattern in _REFUSAL_PATTERNS)


# ------------------------------------------------------------------
# Output object required by evaluator
# ------------------------------------------------------------------

@dataclass
class GeneratedAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


# ------------------------------------------------------------------
# Groq payload adapter
# ------------------------------------------------------------------

class _AsPayload:
    """
    Converts evaluator context objects:

        result.text

    into the payload shape expected by HHGoa's Groq client:

        result.payload["text_en"]
    """

    def __init__(self, text: str):
        self.payload = {
            "text_en": text or ""
        }


# ------------------------------------------------------------------
# Retrieval score helpers
# ------------------------------------------------------------------

def _top_scores(results) -> list[float]:
    return [
        float(getattr(result, "score", 0.0))
        for result in results
    ]


def _top1_score(results) -> float | None:
    if not results:
        return None

    return getattr(results[0], "score", None)


def _score_margin(results) -> float:
    scores = _top_scores(results)

    if len(scores) < 2:
        return 1.0

    return scores[0] - scores[1]


# ------------------------------------------------------------------
# Main evaluator interface
# ------------------------------------------------------------------

def generate_answer(
    query: str,
    results: list
) -> GeneratedAnswer:

    top1 = _top1_score(results)
    margin = _score_margin(results)

    print(
        f"[EVAL DEBUG] query={query!r}\n"
        f"  scores={[round(x, 4) for x in _top_scores(results)]}\n"
        f"  top1={top1}\n"
        f"  margin={margin:.4f}\n"
        f"  direct_threshold={DIRECT_THRESHOLD}\n"
        f"  groq_threshold={GROQ_THRESHOLD}"
    )

    # ==============================================================
    # NO RETRIEVAL
    # ==============================================================

    if not results or top1 is None:

        return GeneratedAnswer(
            text=(
                "I don't have enough information in the "
                "provided context to answer this question."
            ),
            grounded=False,
            generation_ms=0.0,
            model="hhgoa-retrieval-abstain",
        )

    # ==============================================================
    # CASE 1: VERY STRONG RETRIEVAL
    # ==============================================================

    if top1 >= DIRECT_THRESHOLD:

        t0 = time.perf_counter()

        answer_text = results[0].text or ""

        generation_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        return GeneratedAnswer(
            text=answer_text,
            grounded=True,
            generation_ms=generation_ms,
            model="hhgoa-retrieval-direct",
        )

    # ==============================================================
    # CASE 2: MEDIUM RETRIEVAL
    #
    # Let Groq synthesize an answer, but only when retrieval is
    # sufficiently strong.
    # ==============================================================

    if top1 >= GROQ_THRESHOLD:

        # If the retrieval is extremely ambiguous, do not trust it
        # unless the absolute score is very high.
        if (
            margin < AMBIGUOUS_MARGIN
            and top1 < AMBIGUOUS_SCORE
        ):
            return GeneratedAnswer(
                text=(
                    "I don't have enough information in the "
                    "provided context to answer this question."
                ),
                grounded=False,
                generation_ms=0.0,
                model="hhgoa-retrieval-abstain",
            )

        shimmed_results = [
            _AsPayload(getattr(result, "text", ""))
            for result in results
        ]

        t0 = time.perf_counter()

        groq_result = call_groq(
            query,
            shimmed_results,
        )

        generation_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        answer_text = (
            groq_result.get("answer", "") or ""
        )

        # Groq refusal means no grounded answer.
        if _looks_like_refusal(answer_text):

            return GeneratedAnswer(
                text=answer_text,
                grounded=False,
                generation_ms=generation_ms,
                model=settings.GROQ_MODEL,
            )

        return GeneratedAnswer(
            text=answer_text,
            grounded=True,
            generation_ms=generation_ms,
            model=settings.GROQ_MODEL,
        )

    # ==============================================================
    # CASE 3: WEAK RETRIEVAL
    #
    # Important: do NOT allow Groq to turn weak retrieval into
    # an apparently knowledgeable answer.
    # ==============================================================

    return GeneratedAnswer(
        text=(
            "I don't have enough information in the "
            "provided context to answer this question."
        ),
        grounded=False,
        generation_ms=0.0,
        model="hhgoa-retrieval-abstain",
    )