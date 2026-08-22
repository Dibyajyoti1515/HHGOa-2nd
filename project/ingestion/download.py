"""
project/ingestion/download.py

Dataset download + memory-safe Parquet reading.

The source dataset can be several GB, so we NEVER load the complete
Parquet file into memory.

The reader yields pandas DataFrames one batch at a time.
"""

import os
from typing import Iterator

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from project.logging_system.logger import get_logger

logger = get_logger(__name__)

DEFAULT_HF_REPO_ID = "ai4bharat/MSMARCO-XI"
DEFAULT_HF_FILENAME = "train/hintrain.parquet"


def stream_read_parquet(
    path: str,
    batch_size: int = 5000,
) -> Iterator[pd.DataFrame]:
    """
    Stream a Parquet file batch-by-batch.

    IMPORTANT:
    This function intentionally yields one DataFrame at a time.
    It does NOT collect all batches into a list.

    Previous implementation did:

        batches = [batch.to_pandas() for batch in ...]
        return pd.concat(batches)

    which caused the entire 3.7 GB dataset to accumulate in RAM.
    """

    logger.info(
        "Opening parquet for streaming: %s | batch_size=%s",
        path,
        f"{batch_size:,}",
    )

    pf = pq.ParquetFile(path)

    total_rows = pf.metadata.num_rows

    logger.info(
        "Parquet contains approximately %s rows",
        f"{total_rows:,}",
    )

    for batch_number, batch in enumerate(
        pf.iter_batches(batch_size=batch_size),
        start=1,
    ):
        df = batch.to_pandas()

        logger.info(
            "Read parquet batch %s | rows=%s",
            batch_number,
            f"{len(df):,}",
        )

        yield df

        # Explicitly release the local DataFrame reference before
        # moving to the next batch.
        del df


def download_language_file(repo_id: str, filename: str) -> str:
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
    )


def ensure_local_dataset(
    path: str,
    repo_id: str = DEFAULT_HF_REPO_ID,
    filename: str = DEFAULT_HF_FILENAME,
    local_dir: str = None,
) -> str:
    """
    Ensure the source parquet exists locally.

    If it exists, no network request/download is performed.

    Otherwise it is downloaded from Hugging Face.
    """

    if os.path.exists(path):
        logger.info(
            "Source dataset already present: %s",
            path,
        )
        return path

    if local_dir is None:
        normalized_path = path.replace("\\", "/")

        if normalized_path.endswith(filename):
            local_dir = (
                path[: -len(filename)].rstrip("/\\")
                or "."
            )
        else:
            local_dir = os.path.dirname(path) or "."

    os.makedirs(local_dir, exist_ok=True)

    logger.info(
        "Source dataset not found at %s -- downloading %s/%s "
        "from Hugging Face into %s (~3.7GB)",
        path,
        repo_id,
        filename,
        local_dir,
    )

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir,
    )

    logger.info(
        "Download complete: %s",
        downloaded_path,
    )

    return path if os.path.exists(path) else downloaded_path