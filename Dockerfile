# ============================================================
# Backend image — FastAPI + voice pipeline (STT/translation/
# guardrails/retrieval/Groq/TTS) + ingestion CLI.
#
# Build context = repo root (E:\HHGoa). This image is used by
# BOTH the "backend" service (runs the API) and the "ingestion"
# service (runs project/scripts/run_ingestion.py) in
# docker-compose.yml, so both stay in lockstep on dependencies.
# ============================================================
FROM python:3.11-slim AS base
# NOTE: assumed Python 3.11 (matches your mlenv naming convention but
# wasn't confirmed from the repo). If your venv uses a different
# version, change this line — nothing else in the image depends on it.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# build-essential: some of sentence-transformers/qdrant-client's deps
# need a compiler on first install; curl: debugging convenience only.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps first (separate layer, cached across code changes) ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# ---- Application code ----
# Matches your actual repo layout: "project/" holds the backend
# package, "retrieval/" and "scripts/" also live at repo root
# alongside it (see your directory listing) and must be copied too,
# since project.api.app imports from the root-level retrieval package.
COPY project/ ./project/
COPY retrieval/ ./retrieval/
COPY scripts/ ./scripts/

# ---- Docker helper scripts (wait-for-qdrant, entrypoints) ----
COPY docker/ ./docker/
RUN chmod +x ./docker/entrypoint-backend.sh ./docker/entrypoint-ingestion.sh

# ---- Writable dirs for logs/state, created here so bind/volume
# mounts have somewhere to land even before the app writes anything ----
RUN mkdir -p /app/state/logs /app/state/tts_out /app/fastembed_cache

EXPOSE 8000

# Default = run the API. The ingestion service in docker-compose.yml
# overrides both entrypoint and command to run the ingestion CLI instead
# using this SAME image, so ingestion and serving never drift apart.
ENTRYPOINT ["./docker/entrypoint-backend.sh"]