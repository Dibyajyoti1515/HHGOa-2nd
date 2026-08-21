"""
ingestion/download.py

Ported from Cell 2 of Update_version_2.ipynb ("Download + stream-read helpers").
Logic unchanged.
"""

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def stream_read_parquet(path: str, batch_size: int = 5000) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    batches = [batch.to_pandas() for batch in pf.iter_batches(batch_size=batch_size)]
    return pd.concat(batches, ignore_index=True)


def download_language_file(repo_id: str, filename: str) -> str:
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
