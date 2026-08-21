#!/usr/bin/env bash
set -e

python /app/docker/wait_for_qdrant.py

exec uvicorn project.api.app:app --host 0.0.0.0 --port 8000