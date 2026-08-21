"""
ingestion/embeddings/sparse_model.py

Sparse (BM25) embedding model loader via fastembed. Ported from the
notebook cells that load Qdrant/bm25 for hybrid retrieval (the blocks
run after Cell 19: "sparse_model = SparseTextEmbedding(model_name=
'Qdrant/bm25', cache_dir='./fastembed_cache')").
"""

from fastembed import SparseTextEmbedding

from project.config.settings import settings

_sparse_model = None


def get_sparse_model() -> SparseTextEmbedding:
    """Lazily load and cache the sparse (BM25) embedding model."""
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(
            model_name=settings.SPARSE_MODEL,
            cache_dir="./fastembed_cache",
        )
    return _sparse_model
