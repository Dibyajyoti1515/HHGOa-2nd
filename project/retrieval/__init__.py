"""
retrieval/__init__.py

Exposes all 4 clean retrieval functions plus a small mode dispatcher.

The dispatcher is orchestration only (which function to call for a given
RETRIEVAL_MODE string) — it introduces no scoring/fusion logic of its
own. Per CLAUDE.md Decision #1: hybrid_weighted is the production
default and the only user-facing mode; dense / hybrid_rrf / rrf_tunable
stay selectable via RETRIEVAL_MODE for internal comparison only.
"""
from __future__ import annotations

from typing import Callable, Optional

from config.settings import RETRIEVAL_MODE
from retrieval.dense_retriever import retrieve_top_k_dense_only
from retrieval.sparse_retriever import retrieve_top_k_sparse_only
from retrieval.hybrid_rrf_retriever import retrieve_top_k_hybrid_only
from retrieval.hybrid_weighted_retriever import retrieve_top_k_hybrid_weighted
from retrieval.rrf_tunable_retriever import retrieve_top_k_rrf_tunable

__all__ = [
    "retrieve_top_k_dense_only",
    "retrieve_top_k_sparse_only",
    "retrieve_top_k_hybrid_only",
    "retrieve_top_k_hybrid_weighted",
    "retrieve_top_k_rrf_tunable",
    "get_retriever",
]

# mode name -> retrieval function. "hybrid_weighted" is the production
# default (Decision #1). The others exist for comparison only and are
# not meant to be exposed as user-facing options.
_MODES = {
    "dense": retrieve_top_k_dense_only,
    "sparse": retrieve_top_k_sparse_only,
    "hybrid_rrf": retrieve_top_k_hybrid_only,
    "hybrid_weighted": retrieve_top_k_hybrid_weighted,
    "rrf_tunable": retrieve_top_k_rrf_tunable,
}


def get_retriever(mode: Optional[str] = None) -> Callable:
    """Return the retrieval function for `mode` (defaults to settings.RETRIEVAL_MODE).

    Raises ValueError on an unknown mode rather than silently falling back,
    so a typo'd env var fails loudly instead of quietly changing behavior.
    """
    mode = mode or RETRIEVAL_MODE
    try:
        return _MODES[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unknown RETRIEVAL_MODE {mode!r}. Valid options: {sorted(_MODES)}"
        ) from exc
