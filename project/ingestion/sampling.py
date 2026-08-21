"""
ingestion/sampling.py

Ported from Cell 3 of Update_version_2.ipynb ("Stratified subsample").
QUERY_TYPE_DIST now comes from config/settings.py (Cell 1's constant)
instead of a module-level global -- everything else is unchanged.
"""

import pandas as pd

from project.config.settings import settings


def stratified_sample(df: pd.DataFrame, n_total: int, seed: int = 42) -> pd.DataFrame:
    parts = []
    for qtype, frac in settings.QUERY_TYPE_DIST.items():
        n = int(round(n_total * frac))
        pool = df[df["query_type"] == qtype]
        if len(pool) == 0:
            continue
        n = min(n, len(pool))
        parts.append(pool.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True) if parts else df.sample(
        n=min(n_total, len(df)), random_state=seed
    )
