"""
retrieval/hybrid_weighted_retriever.py

Ported from Cell 39 of Update_version_2.ipynb -- retrieve_top_k_hybrid_weighted.

THIS IS THE PRODUCTION DEFAULT (settings.RETRIEVAL_MODE=hybrid_weighted),
chosen because it had the best hit_rate@5 (0.9487) in the notebook's own
Cell 48 benchmark, across 39 queries with valid ground truth. Manually
min-max normalizes dense and sparse candidate scores, then combines them
with a weighted sum (default 0.75 dense / 0.25 sparse, both configurable
via config/settings.py). Logic unchanged from the notebook.
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


def _normalize(points):
    scores = [p.score for p in points]
    if not scores:
        return {}
    lo, hi = min(scores), max(scores)
    rng = (hi - lo) or 1.0
    return {p.payload.get("passage_id"): (p.score - lo) / rng for p in points}


def retrieve_top_k_hybrid_weighted(query_text: str, top_k: int = None,
                                    dense_weight: float = None, sparse_weight: float = None,
                                    candidate_pool: int = None,
                                    model=None, sparse_model=None, qdrant_client=None) -> dict:
    top_k = top_k or settings.TOP_K
    dense_weight = settings.HYBRID_DENSE_WEIGHT if dense_weight is None else dense_weight
    sparse_weight = settings.HYBRID_SPARSE_WEIGHT if sparse_weight is None else sparse_weight
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

    dense_norm = _normalize(dense_resp.points)
    sparse_norm = _normalize(sparse_resp.points)

    point_lookup = {p.payload.get("passage_id"): p for p in dense_resp.points}
    for p in sparse_resp.points:
        point_lookup.setdefault(p.payload.get("passage_id"), p)

    all_ids = set(dense_norm) | set(sparse_norm)
    fused_scores = {
        pid: dense_weight * dense_norm.get(pid, 0.0) + sparse_weight * sparse_norm.get(pid, 0.0)
        for pid in all_ids
    }

    ranked_ids = sorted(fused_scores, key=lambda pid: fused_scores[pid], reverse=True)[:top_k]

    results = []
    for pid in ranked_ids:
        p = point_lookup[pid]
        p.score = fused_scores[pid]
        results.append(p)

    t3 = time.perf_counter()

    result = {
        "results": results,
        "embedding_ms": (t1 - t0) * 1000,
        "qdrant_ms": (t2 - t1) * 1000,
        "fusion_ms": (t3 - t2) * 1000,
        "dedup_ms": 0.0,
        "total_ms": (t3 - t0) * 1000,
    }
    logger.debug(
        "hybrid_weighted retrieval: query=%r top_k=%s total_ms=%.2f top1_score=%s",
        query_text, top_k, result["total_ms"], results[0].score if results else None,
    )
    return result
