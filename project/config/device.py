"""
project/config/device.py

GPU/CPU device resolution for local ML models. This only matters for
the dense embedding model (sentence-transformers / bge-small-en-v1.5)
-- Sarvam STT/TTS, Groq, and Qdrant's own search are remote APIs or
CPU-side ANN search and are unaffected by this.

Controlled by the DEVICE env var:
    "auto" (default) -> "cuda" if a GPU is visible to the container, else "cpu"
    "cuda"            -> force cuda; raises loudly if no GPU is visible
                          (use this if you want a hard failure instead of a
                          silent, much-slower fallback to CPU)
    "cpu"             -> force cpu even if a GPU is present

Call resolve_device() once and pass the result to
SentenceTransformer(..., device=...). Result is cached after the
first call (device availability doesn't change mid-process).
"""

import os

from project.logging_system.logger import get_logger

logger = get_logger(__name__)

_resolved_device = None


def resolve_device() -> str:
    global _resolved_device
    if _resolved_device is not None:
        return _resolved_device

    requested = os.environ.get("DEVICE", "auto").strip().lower()

    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "DEVICE=cuda was requested but no CUDA GPU is visible to this "
                "process. If running in Docker, check that the container was "
                "started with GPU access (docker-compose.gpu.yml) and that the "
                "NVIDIA Container Toolkit is installed on the host."
            )
        device = "cuda"
    elif requested == "cpu":
        device = "cpu"
    else:  # "auto" or anything unrecognized
        device = "cuda" if cuda_available else "cpu"

    if device == "cuda":
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown GPU"
        logger.info("Device resolved: cuda (%s)", gpu_name)
    else:
        logger.info("Device resolved: cpu (DEVICE=%s, cuda_available=%s)", requested, cuda_available)

    _resolved_device = device
    return device