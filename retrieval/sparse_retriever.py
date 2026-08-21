"""
retrieval/sparse_retriever.py

Ported from Cell 26-DEFS of Update_version_2.ipynb -- retrieve_top_k_sparse_only.
"""

import time

from qdrant_client.models import SparseVector

from project.config.settings import settings
from project.ingestion.embeddings.sparse_model import get_sparse_model
from project.ingestion.qdrant_setup import get_qdrant_client
from project.logging_system.logger import get_logger

logger = get_logger(__name__)


def retrieve_top_k_sparse_only(query_text: str, top_k: int = None, sparse_model=None, qdrant_client=None) -> dict:
    top_k = top_k or settings.TOP_K
    sparse_model = sparse_model or get_sparse_model()
    qdrant_client = qdrant_client or get_qdrant_client()

    t0 = time.perf_counter()
    svec = list(sparse_model.embed([query_text]))[0]
    t1 = time.perf_counter()

    response = qdrant_client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=SparseVector(indices=svec.indices.tolist(), values=svec.values.tolist()),
        using="sparse",
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
    logger.debug("sparse retrieval: query=%r top_k=%s total_ms=%.2f", query_text, top_k, result["total_ms"])
    return result
