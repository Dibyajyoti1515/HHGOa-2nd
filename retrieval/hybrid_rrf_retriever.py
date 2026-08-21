"""
retrieval/hybrid_rrf_retriever.py

Ported from Cell 26-DEFS of Update_version_2.ipynb -- retrieve_top_k_hybrid_only.
Uses Qdrant's native RRF fusion (fixed internal k) over a dense+sparse
prefetch (limit 20 each). Not the production default -- see
hybrid_weighted_retriever.py -- but kept available (selectable via
RETRIEVAL_MODE=hybrid_rrf) since it scored close behind in the
notebook's own benchmark and is notably faster.
"""

import time

from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion

from project.config.settings import settings
from project.ingestion.embeddings.dense_model import get_dense_model
from project.ingestion.embeddings.sparse_model import get_sparse_model
from project.ingestion.qdrant_setup import get_qdrant_client
from project.logging_system.logger import get_logger
from retrieval.common import build_prefixed_query

logger = get_logger(__name__)


def retrieve_top_k_hybrid_only(query_text: str, top_k: int = None, model=None, sparse_model=None,
                                qdrant_client=None) -> dict:
    top_k = top_k or settings.TOP_K
    model = model or get_dense_model()
    sparse_model = sparse_model or get_sparse_model()
    qdrant_client = qdrant_client or get_qdrant_client()

    t0 = time.perf_counter()
    prefixed = build_prefixed_query(query_text)
    dvec = model.encode(prefixed, normalize_embeddings=True).tolist()
    svec = list(sparse_model.embed([query_text]))[0]
    t1 = time.perf_counter()

    response = qdrant_client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        prefetch=[
            Prefetch(query=dvec, using="dense", limit=20),
            Prefetch(
                query=SparseVector(indices=svec.indices.tolist(), values=svec.values.tolist()),
                using="sparse",
                limit=20,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
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
    logger.debug("hybrid_rrf retrieval: query=%r top_k=%s total_ms=%.2f", query_text, top_k, result["total_ms"])
    return result
