"""
llm_fallback/groq_client.py

Groq LLM fallback. Used when the top1 retrieval confidence score is
below settings.CONFIDENCE_THRESHOLD (CLAUDE.md decision #3): instead
of reading the raw retrieved passage aloud, Groq composes a direct
answer to the query from the retrieved context.

NEW module -- not from the notebook. Ported from the user's own draft
app.py, wired against this project's config/logging.
"""

import time

from groq import Groq

from project.config.settings import settings
from project.logging_system.logger import get_logger

logger = get_logger(__name__)

_groq_client = None


def get_groq_client() -> Groq:
    """Lazily create and cache the Groq client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    return _groq_client


def _build_context(results) -> str:
    """Join retrieved passage texts (payload['text_en']) into a context block."""
    passages = []
    for r in results:
        text = r.payload.get("text_en", "")
        if text:
            passages.append(text)
    return "\n\n".join(passages)


def call_groq(query: str, results, model: str = None) -> dict:
    """
    Compose an answer to `query` using `results` (a list of Qdrant
    ScoredPoint objects from retrieval) as context.

    Returns:
        {"answer": <str>, "groq_ms": <float>}
    """
    model = model or settings.GROQ_MODEL
    client = get_groq_client()
    context = _build_context(results)

    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict retrieval-grounded QA system.\n\n"
            "Your ONLY source of information is the provided context.\n"
            "Do NOT use your own world knowledge.\n"
            "Do NOT infer an answer merely because the context is related "
            "to the question.\n\n"
            "First determine whether the context explicitly contains "
            "enough information to answer the question.\n\n"
            "If the context does NOT contain enough information, respond "
            "with exactly:\n"
            "NOT_SUPPORTED\n\n"
            "If the context DOES contain enough information, answer "
            "concisely using ONLY information present in the context."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
        )
        answer = response.choices[0].message.content.strip()
    except Exception:
        logger.exception("Groq fallback call failed for query=%r", query)
        raise
    t1 = time.perf_counter()

    groq_ms = (t1 - t0) * 1000
    logger.info("Groq fallback used: query=%r model=%s groq_ms=%.2f", query, model, groq_ms)

    return {
        "answer": answer,
        "groq_ms": groq_ms,
    }