"""
retrieval/dense_retriever.py

Ported from Cell 26-DEFS of Update_version_2.ipynb -- retrieve_top_k_dense_only.

Note: the notebook also defines an identical retrieve_top_k_silent, with
its own comment: "same as dense_only -- kept separate name to match your
original benchmark code." It is not ported as a separate file since it
is a byte-for-byte duplicate of this function.
"""

import time

from project.config.settings import settings
from project.ingestion.embeddings.dense_model import get_dense_model
from project.ingestion.qdrant_setup import get_qdrant_client
from project.logging_system.logger import get_logger
from project.retrieval.common import build_prefixed_query

logger = get_logger(__name__)


def retrieve_top_k_dense_only(query_text: str, top_k: int = None, model=None, qdrant_client=None) -> dict:
    top_k = top_k or settings.TOP_K
    model = model or get_dense_model()
    qdrant_client = qdrant_client or get_qdrant_client()

    t0 = time.perf_counter()
    prefixed = build_prefixed_query(query_text)
    qvec = model.encode(prefixed, normalize_embeddings=True).tolist()
    t1 = time.perf_counter()

    response = qdrant_client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=qvec,
        using="dense",
        limit=top_k,
    )
    t2 = time.perf_counter()

    result = {
        "results": response.points,
        "embedding_ms": (t1 - t0) * 1000,
        "qdrant_ms": (t2 - t1) * 1000,
        "dedup_ms": 0.0,
        "total_ms": (t2 - t0) * 1000,
    }
    logger.debug("dense retrieval: query=%r top_k=%s total_ms=%.2f", query_text, top_k, result["total_ms"])
    return result