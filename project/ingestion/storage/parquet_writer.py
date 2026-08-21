"""
ingestion/storage/parquet_writer.py

Ported from Cell 9 of Update_version_2.ipynb -- write_unified_file.
UNIFIED_FILE now comes from config/settings.py; print() replaced with
the central logger. Logic unchanged.
"""

import pandas as pd

from project.config.settings import settings
from project.logging_system.logger import get_logger

logger = get_logger(__name__)


def write_unified_file(df: pd.DataFrame, path: str = None):
    if path is None:
        path = settings.UNIFIED_FILE

    df.to_parquet(path, index=False)
    logger.info("Archive written: %s rows -> %s", f"{len(df):,}", path)
