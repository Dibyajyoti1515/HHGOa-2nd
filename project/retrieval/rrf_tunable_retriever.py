"""
retrieval/rrf_tunable_retriever.py

Ported from Cell 45 of Update_version_2.ipynb -- retrieve_top_k_rrf_tunable.
Manual Reciprocal Rank Fusion with a tunable k. The notebook benchmarked
k in {5, 10, 30, 60, 100} in Cell 46/48 -- hybrid_weighted still won on
hit_rate@5 in every case, so this is kept available (selectable via
RETRIEVAL_MODE=rrf_tunable) but is NOT the production default.
"""

import time

from qdrant_client.models import SparseVector

from project.config.settings import settings
from project.ingestion.embeddings.dense_model import get_dense_model
from project.ingestion.embeddings.sparse_model import get_sparse_model
from project.ingestion.qdrant_setup import get_qdrant_client
from project.logging_system.logger import get_logger
from retrieval.common import build_prefixed_query

logger = get_logger(__name__)


def retrieve_top_k_rrf_tunable(query_text: str, top_k: int = None, k: int = 60,
                                candidate_pool: int = None,
                                model=None, sparse_model=None, qdrant_client=None) -> dict:
    top_k = top_k or settings.TOP_K
    candidate_pool = candidate_pool or settings.HYBRID_CANDIDATE_POOL
    model = model or get_dense_model()
    sparse_model = sparse_model or get_sparse_model()
    qdrant_client = qdrant_client or get_qdrant_client()

    t0 = time.perf_counter()
    prefixed = build_prefixed_query(query_text)
    dvec = model.encode(prefixed, normalize_embeddings=True).tolist()
    svec = list(sparse_model.embed([query_text]))[0]
    t1 = time.perf_counter()

    dense_resp = qdrant_client.query_points(
        collection_name=settings.QDRANT_COLLECTION, query=dvec, using="dense", limit=candidate_pool
    )
    sparse_resp = qdrant_client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=SparseVector(indices=svec.indices.tolist(), values=svec.values.tolist()),
        using="sparse", limit=candidate_pool
    )
    t2 = time.perf_counter()

    rrf_scores = {}
    point_lookup = {}
    for rank, p in enumerate(dense_resp.points, 1):
        pid = p.payload.get("passage_id")
        rrf_scores[pid] = rrf_scores.get(pid, 0) + 1 / (k + rank)
        point_lookup[pid] = p
    for rank, p in enumerate(sparse_resp.points, 1):
        pid = p.payload.get("passage_id")
        rrf_scores[pid] = rrf_scores.get(pid, 0) + 1 / (k + rank)
        point_lookup.setdefault(pid, p)

    ranked = sorted(rrf_scores, key=lambda pid: rrf_scores[pid], reverse=True)[:top_k]
    results = []
    for pid in ranked:
        p = point_lookup[pid]
        p.score = rrf_scores[pid]
        results.append(p)

    t3 = time.perf_counter()

    result = {
        "results": results,
        "embedding_ms": (t1 - t0) * 1000,
        "qdrant_ms": (t2 - t1) * 1000,
        "fusion_ms": (t3 - t2) * 1000,
        "total_ms": (t3 - t0) * 1000,
    }
    logger.debug("rrf_tunable retrieval: query=%r k=%s top_k=%s total_ms=%.2f", query_text, k, top_k, result["total_ms"])
    return result
