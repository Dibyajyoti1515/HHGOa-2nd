"""
project/ingestion/embeddings/dense_model.py

Dense embedding model loader (bge-small-en-v1.5). Used by both
ingestion (chunking/embedding the corpus) and retrieval (embedding
queries), so it stays in one place.

GPU/CPU: device is resolved once via project.config.device.resolve_device()
(checks torch.cuda.is_available(), or a forced DEVICE env var) and passed
to SentenceTransformer -- runs on GPU automatically when one is visible
to the process (e.g. via docker-compose.gpu.yml), falls back to CPU
otherwise. No code path needs to know or care which one it got.
"""

from sentence_transformers import SentenceTransformer

from project.config.settings import settings
from project.config.device import resolve_device
from project.logging_system.logger import get_logger

logger = get_logger(__name__)

_dense_model = None


def get_dense_model() -> SentenceTransformer:
    """Lazily load and cache the dense embedding model, on GPU if available."""
    global _dense_model
    if _dense_model is None:
        device = resolve_device()
        _dense_model = SentenceTransformer(settings.EMBED_MODEL, device=device)  # <-- added device=
        logger.info("Dense model '%s' loaded on device=%s", settings.EMBED_MODEL, device)
    return _dense_model