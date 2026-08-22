"""
project/ingestion/sampling.py

Memory-safe stratified sampling.

Instead of loading the complete MS MARCO-XI dataset,
we maintain a small reservoir for each query_type while
streaming through the Parquet file.
"""

from typing import Iterable

import pandas as pd

from project.config.settings import settings
from project.logging_system.logger import get_logger

logger = get_logger(__name__)


def stratified_sample(
    df: pd.DataFrame,
    n_total: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Original DataFrame-based stratified sampler.

    Kept for compatibility with existing code.
    """

    parts = []

    for qtype, frac in settings.QUERY_TYPE_DIST.items():
        n = int(round(n_total * frac))

        pool = df[df["query_type"] == qtype]

        if len(pool) == 0:
            continue

        n = min(n, len(pool))

        parts.append(
            pool.sample(
                n=n,
                random_state=seed,
            )
        )

    if parts:
        return pd.concat(
            parts,
            ignore_index=True,
        )

    return df.sample(
        n=min(n_total, len(df)),
        random_state=seed,
    )


def _reservoir_add(
    reservoir: list,
    row: dict,
    capacity: int,
    seen_count: int,
    rng,
):
    """
    Add one row to a reservoir using standard reservoir sampling.

    Memory usage is bounded by `capacity`.
    """

    if len(reservoir) < capacity:
        reservoir.append(row)
        return

    replacement_index = rng.randint(0, seen_count)

    if replacement_index < capacity:
        reservoir[replacement_index] = row


def stratified_sample_stream(
    batches: Iterable[pd.DataFrame],
    n_total: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Memory-safe stratified sampling over streamed DataFrame batches.

    Only approximately n_total rows are retained in memory,
    regardless of the size of the source dataset.
    """

    import random

    rng = random.Random(seed)

    # Calculate target sample count for every query type.
    targets = {
        qtype: int(round(n_total * frac))
        for qtype, frac in settings.QUERY_TYPE_DIST.items()
    }

    reservoirs = {
        qtype: []
        for qtype in targets
    }

    seen_counts = {
        qtype: 0
        for qtype in targets
    }

    total_seen = 0

    logger.info(
        "Starting streaming stratified sampling | target=%s",
        f"{n_total:,}",
    )

    for batch_number, batch in enumerate(batches, start=1):

        if "query_type" not in batch.columns:
            raise ValueError(
                "Source parquet does not contain 'query_type'"
            )

        for row in batch.itertuples(index=False):

            row_dict = row._asdict()

            qtype = row_dict.get("query_type")

            if qtype not in targets:
                continue

            seen_counts[qtype] += 1
            total_seen += 1

            _reservoir_add(
                reservoir=reservoirs[qtype],
                row=row_dict,
                capacity=targets[qtype],
                seen_count=seen_counts[qtype],
                rng=rng,
            )

        logger.info(
            "Sampling batch %s | rows scanned=%s | retained=%s",
            batch_number,
            f"{len(batch):,}",
            sum(len(v) for v in reservoirs.values()),
        )

        del batch

    parts = []

    for qtype, reservoir in reservoirs.items():

        if not reservoir:
            continue

        parts.append(
            pd.DataFrame(reservoir)
        )

    if not parts:
        logger.warning(
            "No rows matched QUERY_TYPE_DIST. "
            "Returning empty DataFrame."
        )
        return pd.DataFrame()

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    # Shuffle the final sample so query types are mixed.
    result = result.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    logger.info(
        "Streaming sampling complete | scanned=%s | selected=%s",
        f"{total_seen:,}",
        f"{len(result):,}",
    )

    return result