"""
ingestion/embeddings/dense_model.py

Dense embedding model loader. Ported from the hybrid ingestion path
(embed_and_upsert_hybrid, defined after Cell 19 in the notebook) --
this is the version actually used for production ingestion, since the
notebook's earlier dense-only embed_and_upsert (Cell 10 / run in
Cell 17) was superseded once hybrid retrieval was introduced.
"""

from sentence_transformers import SentenceTransformer

from project.config.settings import settings

_dense_model = None


def get_dense_model() -> SentenceTransformer:
    """Lazily load and cache the dense embedding model (bge-small-en-v1.5)."""
    global _dense_model
    if _dense_model is None:
        _dense_model = SentenceTransformer(settings.EMBED_MODEL, local_files_only=True)
    return _dense_model
