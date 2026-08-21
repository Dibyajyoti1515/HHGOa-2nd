"""
ingestion/dedup.py

Ported from Cell 8 of Update_version_2.ipynb ("Dedup with
passage_native-first priority"). DEDUP_THRESHOLD now comes from
config/settings.py instead of a module-level global; the notebook's
two print() progress lines are now routed through the central logger.
Everything else -- the dedup algorithm itself -- is unchanged.
"""

from difflib import SequenceMatcher

import pandas as pd

from project.config.settings import settings
from project.logging_system.logger import get_logger

logger = get_logger(__name__)

STRATEGY_PRIORITY = {"passage_native": 0, "semantic": 1, "fixed_overlap": 2}


def dedupe_chunks(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    if threshold is None:
        threshold = settings.DEDUP_THRESHOLD

    keep_rows = []
    before = len(df)
    for passage_id, group in df.groupby("passage_id"):
        group_sorted = group.copy()
        group_sorted["_priority"] = group_sorted["chunk_strategy"].map(STRATEGY_PRIORITY).fillna(99)
        group_sorted = group_sorted.sort_values("_priority")
        seen_texts = []
        for _, row in group_sorted.iterrows():
            text = row["text_en"] or ""
            if any(SequenceMatcher(None, text, s).ratio() > threshold for s in seen_texts):
                continue
            seen_texts.append(text)
            keep_rows.append(row.drop("_priority"))
    result = pd.DataFrame(keep_rows)
    after = len(result)
    reduction_pct = (before - after) / before * 100 if before else 0.0
    logger.info(
        "Dedup: %s -> %s chunks (%s removed, %.1f%% reduction)",
        f"{before:,}", f"{after:,}", f"{before - after:,}", reduction_pct,
    )
    logger.info("Chunk strategy counts:\n%s", result["chunk_strategy"].value_counts().to_string())
    return result
