"""
project/ingestion/download.py

Ported from Cell 2 of Update_version_2.ipynb ("Download + stream-read
helpers"), plus a new ensure_local_dataset() helper: if the configured
source parquet isn't on disk yet, create its parent directory and
download it from Hugging Face automatically instead of failing.
"""

import os

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from logging_system.logger import get_logger

logger = get_logger(__name__)

# Where the source MS MARCO-XI parquet actually lives on Hugging Face.
# Matches Cell 1's local DATA_DIR ("./MSMARCO-XI/train/hintrain.parquet")
# -- this is that same file's home upstream.
DEFAULT_HF_REPO_ID = "ai4bharat/MSMARCO-XI"
DEFAULT_HF_FILENAME = "train/hintrain.parquet"


def stream_read_parquet(path: str, batch_size: int = 5000) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    batches = [batch.to_pandas() for batch in pf.iter_batches(batch_size=batch_size)]
    return pd.concat(batches, ignore_index=True)


def download_language_file(repo_id: str, filename: str) -> str:
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")


def ensure_local_dataset(
    path: str,
    repo_id: str = DEFAULT_HF_REPO_ID,
    filename: str = DEFAULT_HF_FILENAME,
    local_dir: str = None,
) -> str:
    """
    Make sure a source parquet exists at `path`, downloading it from
    Hugging Face if it doesn't.

    - If `path` already exists: used as-is, nothing downloaded.
    - Otherwise: creates the parent directory (if missing) and downloads
      `filename` from the `repo_id` dataset repo. Hugging Face preserves
      the file's own subpath under whatever `local_dir` you give it
      (e.g. "train/hintrain.parquet"), so `local_dir` is worked out here
      by stripping that subpath off the end of `path`, ensuring the
      download lands at exactly `path` rather than some nested location.

    Returns the path the file actually ended up at. Callers should use
    the return value (not assume `path` itself was written to) in the
    rare case the subpath-stripping heuristic above doesn't apply --
    it then falls back to hf_hub_download's own cache location.
    """
    if os.path.exists(path):
        logger.info("Source dataset already present: %s", path)
        return path

    if local_dir is None:
        if path.replace("\\", "/").endswith(filename):
            local_dir = path[: -len(filename)].rstrip("/\\") or "."
        else:
            local_dir = os.path.dirname(path) or "."

    os.makedirs(local_dir, exist_ok=True)

    logger.info(
        "Source dataset not found at %s -- downloading %s/%s from Hugging "
        "Face into %s (this is a ~3.7GB file; first run will take a while)",
        path, repo_id, filename, local_dir,
    )
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir,
    )
    logger.info("Download complete: %s", downloaded_path)

    return path if os.path.exists(path) else downloaded_path