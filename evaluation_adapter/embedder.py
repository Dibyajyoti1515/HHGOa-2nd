"""
evaluation_adapter/embedder.py

Satisfies rag-local-eval-loop's embedder interface (TARGET_INTERFACE.md):

    embed(texts: list[str]) -> array-like, shape (len(texts), dim)
    embed_one(text: str)     -> array-like, shape (dim,)
    get_model()              -> anything; only the loading side effect matters

Reuses HHGoa's existing dense embedding model
(project.ingestion.embeddings.dense_model.get_dense_model, BAAI/bge-small-en-v1.5,
loaded once and cached at module level there -- calling get_dense_model()
again here does not reload it). No second model is initialized.

Design note -- no query prefix here:
HHGoa's production query path (project/retrieval/dense_retriever.py)
prepends a retrieval-instruction prefix ("Represent this sentence for
searching relevant passages: ") to *queries only*, via
project/retrieval/common.py:build_prefixed_query, before encoding them
for the Qdrant "dense" vector. That prefix is retrieval-strategy-specific,
not a property of the embedding model itself.

rag-local-eval-loop calls this module's embed()/embed_one() for BOTH
sides of its throwaway FAISS index: passages (eval/index_build.py calls
embed() on candidate passage chunks) and queries (eval/pipeline.py calls
embed_one() on the query text). Applying the query-only prefix here would
mean passages get the "searching relevant passages" prefix too, which is
wrong for text that isn't a search query. So this adapter intentionally
encodes everything the same way, unprefixed -- symmetric embedding is
what a single embed()/embed_one() pair calling into a shared index can
correctly support. This does not touch or change the production
retrieval path in project/retrieval/, which keeps prefixing queries as
before.

Both vectors are L2-normalized (normalize_embeddings=True, matching
project/retrieval/dense_retriever.py), and the eval suite's FAISS index
is built with METRIC_INNER_PRODUCT (see eval/index_build.py) -- so scores
there are cosine similarities, the same semantics as the scores Qdrant
returns from the production "dense" vector.
"""
from __future__ import annotations

import numpy as np

from project.ingestion.embeddings.dense_model import get_dense_model


def get_model():
    """Loads (or returns the already-cached) HHGoa dense embedding model."""
    return get_dense_model()


def embed_one(text: str) -> np.ndarray:
    model = get_dense_model()
    vec = model.encode(text, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def embed(texts: list[str]) -> np.ndarray:
    model = get_dense_model()
    if not texts:
        dim = embed_one("dimension probe").shape[-1]
        return np.zeros((0, dim), dtype=np.float32)
    vecs = model.encode(list(texts), normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)