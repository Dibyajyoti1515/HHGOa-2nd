"""
ingestion/pipeline.py

Memory-safe ingestion orchestrator.

The large source Parquet is streamed in batches.

Only the configured stratified sample is retained in memory.
The existing downstream notebook pipeline then operates on
that bounded sample.
"""

import gc

import pandas as pd

from project.config.settings import settings
from project.logging_system.logger import get_logger

from project.ingestion.download import (
    ensure_local_dataset,
    stream_read_parquet,
)

from project.ingestion.sampling import (
    stratified_sample_stream,
)

from project.ingestion.explode import (
    explode_msmarco_xi,
)

from project.ingestion.preprocess import (
    preprocess_corpus,
)

from project.ingestion.enrich import (
    enrich,
)

from project.ingestion.chunking.native import (
    chunk_passage_native,
)

from project.ingestion.chunking.fixed_overlap import (
    chunk_fixed_overlap,
)

from project.ingestion.chunking.semantic import (
    chunk_semantic,
)

from project.ingestion.chunking.tagging import (
    tag_metadata_aware,
)

from project.ingestion.dedup import (
    dedupe_chunks,
)

from project.ingestion.storage.parquet_writer import (
    write_unified_file,
)

from project.ingestion.storage.sqlite_writer import (
    build_native_lookup,
)

from project.ingestion.embeddings.dense_model import (
    get_dense_model,
)

from project.ingestion.embeddings.sparse_model import (
    get_sparse_model,
)

from project.ingestion.qdrant_setup import (
    get_qdrant_client,
    ensure_collection,
    ensure_payload_indexes,
)

from project.ingestion.upsert import (
    embed_dense,
    embed_sparse,
    build_points,
    upsert_batch,
)


logger = get_logger(__name__)


def run_ingestion(
    data_path: str,
    lang_code: str = "en",
) -> None:

    """
    Full ingestion pipeline.

    Memory strategy:

        Large Parquet
            ↓
        streamed batches
            ↓
        stratified reservoir sample
            ↓
        bounded DataFrame
            ↓
        existing processing pipeline
            ↓
        Qdrant

    This prevents the complete 3.7GB source file from being
    materialized into RAM.
    """

    data_path = ensure_local_dataset(
        data_path
    )

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    parquet_batch_size = getattr(
        settings,
        "INGESTION_BATCH_SIZE",
        1000,
    )

    sample_size = (
        settings.SAMPLE_SIZE_PER_LANG
    )

    logger.info(
        "Ingestion configuration:"
    )

    logger.info(
        "  parquet batch size = %s",
        f"{parquet_batch_size:,}",
    )

    logger.info(
        "  target sample size = %s",
        f"{sample_size:,}",
    )

    # ---------------------------------------------------------
    # Cell 11
    #
    # Stream giant parquet and keep only the required sample.
    # ---------------------------------------------------------

    logger.info(
        "Reading source parquet using streaming sampling: %s",
        data_path,
    )

    parquet_batches = stream_read_parquet(
        data_path,
        batch_size=parquet_batch_size,
    )

    df = stratified_sample_stream(
        parquet_batches,
        n_total=sample_size,
        seed=42,
    )

    logger.info(
        "Sample obtained: %s rows",
        f"{len(df):,}",
    )

    if df.empty:
        logger.error(
            "Sampling returned zero rows. "
            "Check QUERY_TYPE_DIST and source data."
        )
        return

    # ---------------------------------------------------------
    # Cell 11 continued
    # ---------------------------------------------------------

    normalized = explode_msmarco_xi(
        df,
        lang_code,
    )

    logger.info(
        "Normalized: %s passage rows",
        f"{len(normalized):,}",
    )

    del df
    gc.collect()

    # ---------------------------------------------------------
    # Cell 12
    # ---------------------------------------------------------

    unified = pd.concat(
        [normalized],
        ignore_index=True,
    )

    del normalized
    gc.collect()

    unified = preprocess_corpus(
        unified
    )

    logger.info(
        "Unified after preprocessing: %s rows",
        f"{len(unified):,}",
    )

    # ---------------------------------------------------------
    # Cell 13
    # ---------------------------------------------------------

    unified = enrich(
        unified
    )

    logger.info(
        "Enrichment done."
    )

    # ---------------------------------------------------------
    # Cell 14
    # ---------------------------------------------------------

    native = chunk_passage_native(
        unified
    )

    fixed = chunk_fixed_overlap(
        unified
    )

    dense_model = get_dense_model()

    semantic = chunk_semantic(
        unified,
        dense_model,
    )

    logger.info(
        "native: %s | fixed: %s | semantic: %s (all pre-dedup)",
        f"{len(native):,}",
        f"{len(fixed):,}",
        f"{len(semantic):,}",
    )

    del unified
    gc.collect()

    all_chunks = pd.concat(
        [
            native,
            fixed,
            semantic,
        ],
        ignore_index=True,
    )

    del native
    del fixed
    del semantic
    gc.collect()

    # ---------------------------------------------------------
    # Cell 15
    # ---------------------------------------------------------

    all_chunks = tag_metadata_aware(
        all_chunks
    )

    all_chunks = dedupe_chunks(
        all_chunks
    )

    gc.collect()

    # ---------------------------------------------------------
    # Cell 16
    # ---------------------------------------------------------

    write_unified_file(
        all_chunks
    )

    build_native_lookup(
        all_chunks
    )

    # ---------------------------------------------------------
    # Cell 17
    # ---------------------------------------------------------

    all_chunks = all_chunks[
        all_chunks["text_en"].notna()
        & (
            all_chunks["text_en"]
            .str.strip()
            != ""
        )
    ].reset_index(drop=True)

    logger.info(
        "Chunks ready for embedding: %s",
        f"{len(all_chunks):,}",
    )

    sparse_model = get_sparse_model()

    client = get_qdrant_client()

    ensure_collection(
        client,
        recreate=True,
    )

    ensure_payload_indexes(
        client
    )

    # ---------------------------------------------------------
    # Embedding + Qdrant
    # ---------------------------------------------------------

    texts = (
        all_chunks[
            "text_en"
        ]
        .tolist()
    )

    dense_vectors = embed_dense(
        texts,
        dense_model,
    )

    sparse_vectors = embed_sparse(
        texts,
        sparse_model,
    )

    points = build_points(
        all_chunks,
        dense_vectors,
        sparse_vectors,
    )

    upsert_batch(
        client,
        points,
    )

    logger.info(
        "Ingestion pipeline complete."
    )

    # ---------------------------------------------------------
    # Final cleanup
    # ---------------------------------------------------------

    del texts
    del dense_vectors
    del sparse_vectors
    del points
    del all_chunks

    gc.collect()