"""
ingestion/storage/sqlite_writer.py

Ported from Cell 9 of Update_version_2.ipynb -- build_native_lookup.
SQLITE_DB now comes from config/settings.py; print() replaced with the
central logger. Logic unchanged.
"""

import sqlite3

import pandas as pd

from project.config.settings import settings
from project.logging_system.logger import get_logger

logger = get_logger(__name__)


def build_native_lookup(df: pd.DataFrame, db_path: str = None):
    if db_path is None:
        db_path = settings.SQLITE_DB

    native_lookup = df.drop_duplicates(subset=["passage_id"])[
        ["passage_id", "query_id", "lang", "text", "query", "answer"]
    ]
    conn = sqlite3.connect(db_path)
    native_lookup.to_sql("passages", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_passage_id ON passages(passage_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_query_id ON passages(query_id)")
    conn.commit()
    conn.close()
    logger.info("SQLite native lookup: %s unique passages -> %s", f"{len(native_lookup):,}", db_path)
