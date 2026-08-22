#!/usr/bin/env bash
set -e

python /app/docker/wait_for_qdrant.py

# If --input points at a parquet that doesn't exist yet, run_ingestion.py
# should call ingestion.download.ensure_local_dataset(data_path) before
# reading it -- that creates the parent dir and pulls the file from
# Hugging Face (ai4bharat/MSMARCO-XI) automatically. See
# project/ingestion/download.py.
#
# ASSUMPTION (unverified against your actual project/scripts/run_ingestion.py,
# GitHub wouldn't let me fetch it): invoked as a module with --input/--lang,
# matching the CLI shape from CLAUDE.md's build-order notes. If your real
# script takes different flags, only this line needs to change --
# everything else (image, deps, wait-for-qdrant) stays as-is.
exec python -m project.scripts.run_ingestion "$@"