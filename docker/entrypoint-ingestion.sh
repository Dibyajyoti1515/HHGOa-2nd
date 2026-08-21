#!/usr/bin/env bash
set -e

python /app/docker/wait_for_qdrant.py

# ASSUMPTION (unverified against your actual project/scripts/run_ingestion.py,
# GitHub wouldn't let me fetch it): invoked as a module with --input/--lang,
# matching the CLI shape from CLAUDE.md's build-order notes. If your real
# script takes different flags, only this line needs to change --
# everything else (image, deps, wait-for-qdrant) stays as-is.
exec python -m project.scripts.run_ingestion "$@"