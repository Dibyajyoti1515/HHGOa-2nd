"""
ingestion/chunking/native.py

Ported from Cell 7 of Update_version_2.ipynb -- chunk_passage_native.
One of three chunking strategies: passage-native (no re-chunking,
just relabeled with a chunk_id). Logic unchanged.
"""

import pandas as pd


def chunk_passage_native(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["chunk_strategy"] = "passage_native"
    out["chunk_id"] = out["passage_id"] + "_native"
    return out
