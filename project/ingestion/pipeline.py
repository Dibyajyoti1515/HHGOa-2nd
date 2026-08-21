"""
ingestion/pipeline.py

Orchestrator. Chains ingestion/* modules in the same order the
notebook ran them (Cells 11 -> 18), for the hybrid (dense+sparse)
production path. English-only per project decision -- the notebook's
multi-language hook (pd.concat([hi_norm]) with a comment about adding
Odia/Telugu "later if needed") was never exercised and is not ported.
"""

import pandas as pd

from project.config.settings import settings
from project.logging_system.logger import get_logger

from project.ingestion.download import stream_read_parquet
from project.ingestion.sampling import stratified_sample
from project.ingestion.explode import explode_msmarco_xi
from project.ingestion.preprocess import preprocess_corpus
from project.ingestion.enrich import enrich
from project.ingestion.chunking.native import chunk_passage_native
from project.ingestion.chunking.fixed_overlap import chunk_fixed_overlap
from project.ingestion.chunking.semantic import chunk_semantic
from project.ingestion.chunking.tagging import tag_metadata_aware
from project.ingestion.dedup import dedupe_chunks
from project.ingestion.storage.parquet_writer import write_unified_file
from project.ingestion.storage.sqlite_writer import build_native_lookup
from project.ingestion.embeddings.dense_model import get_dense_model
from project.ingestion.embeddings.sparse_model import get_sparse_model
from project.ingestion.qdrant_setup import get_qdrant_client, ensure_collection, ensure_payload_indexes
from project.ingestion.upsert import embed_dense, embed_sparse, build_points, upsert_batch

logger = get_logger(__name__)


def run_ingestion(data_path: str, lang_code: str = "en") -> None:
    """
    Full ingestion pipeline, mirroring the notebook's Cells 11 -> 18:
      11: download/read + stratified sample + explode
      12: combine + preprocess
      13: enrich (years + NER)
      14: chunk (all 3 strategies: native, fixed_overlap, semantic)
      15: tag metadata + dedup
      16: write archive (parquet) + SQLite native lookup
      17: embed (dense + sparse) + upsert to Qdrant (hybrid collection)
    """
    # Cell 11
    logger.info("Reading source parquet: %s", data_path)
    df = stream_read_parquet(data_path)
    df = stratified_sample(df, settings.SAMPLE_SIZE_PER_LANG)
    normalized = explode_msmarco_xi(df, lang_code)
    logger.info("Normalized: %s passage rows", f"{len(normalized):,}")

    # Cell 12
    unified = pd.concat([normalized], ignore_index=True)
    unified = preprocess_corpus(unified)
    logger.info("Unified after preprocessing: %s rows", f"{len(unified):,}")

    # Cell 13
    unified = enrich(unified)
    logger.info("Enrichment done.")

    # Cell 14
    native = chunk_passage_native(unified)
    fixed = chunk_fixed_overlap(unified)
    dense_model = get_dense_model()
    semantic = chunk_semantic(unified, dense_model)
    logger.info(
        "native: %s | fixed: %s | semantic: %s (all pre-dedup)",
        f"{len(native):,}", f"{len(fixed):,}", f"{len(semantic):,}",
    )
    all_chunks = pd.concat([native, fixed, semantic], ignore_index=True)

    # Cell 15
    all_chunks = tag_metadata_aware(all_chunks)
    all_chunks = dedupe_chunks(all_chunks)

    # Cell 16
    write_unified_file(all_chunks)
    build_native_lookup(all_chunks)

    # Cell 17 (hybrid embed + upsert)
    all_chunks = all_chunks[
        all_chunks["text_en"].notna() & (all_chunks["text_en"].str.strip() != "")
    ].reset_index(drop=True)

    sparse_model = get_sparse_model()
    client = get_qdrant_client()
    ensure_collection(client, recreate=True)
    ensure_payload_indexes(client)

    texts = all_chunks["text_en"].tolist()
    dense_vectors = embed_dense(texts, dense_model)
    sparse_vectors = embed_sparse(texts, sparse_model)
    points = build_points(all_chunks, dense_vectors, sparse_vectors)
    upsert_batch(client, points)

    logger.info("Ingestion pipeline complete.")
