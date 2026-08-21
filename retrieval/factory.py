"""
retrieval/factory.py

NEW -- not from the notebook. Dispatches to the retrieval strategy
configured via settings.RETRIEVAL_MODE (default: hybrid_weighted)
without callers needing to import a specific retriever module. No
retrieval logic lives here; this only selects among the ported
functions in this package.
"""

from project.config.settings import settings
from retrieval.dense_retriever import retrieve_top_k_dense_only
from retrieval.sparse_retriever import retrieve_top_k_sparse_only
from retrieval.hybrid_rrf_retriever import retrieve_top_k_hybrid_only
from retrieval.hybrid_weighted_retriever import retrieve_top_k_hybrid_weighted
from retrieval.rrf_tunable_retriever import retrieve_top_k_rrf_tunable

RETRIEVAL_FUNCTIONS = {
    "dense": retrieve_top_k_dense_only,
    "sparse": retrieve_top_k_sparse_only,
    "hybrid_rrf": retrieve_top_k_hybrid_only,
    "hybrid_weighted": retrieve_top_k_hybrid_weighted,
    "rrf_tunable": retrieve_top_k_rrf_tunable,
}


def retrieve(query_text: str, mode: str = None, **kwargs) -> dict:
    """
    Run retrieval using the given mode (defaults to settings.RETRIEVAL_MODE,
    i.e. hybrid_weighted in production). Extra kwargs are forwarded to the
    underlying retrieval function (e.g. top_k, or k for rrf_tunable).
    """
    mode = mode or settings.RETRIEVAL_MODE
    if mode not in RETRIEVAL_FUNCTIONS:
        raise ValueError(f"Unknown retrieval mode: {mode!r}. Valid modes: {list(RETRIEVAL_FUNCTIONS)}")
    return RETRIEVAL_FUNCTIONS[mode](query_text, **kwargs)
