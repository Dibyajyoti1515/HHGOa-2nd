"""
project/ingestion/upsert.py

Dense + sparse embedding and Qdrant upsert helpers.
"""

import uuid

import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    SparseVector,
)

from project.config.settings import settings
from project.logging_system.logger import get_logger


logger = get_logger(__name__)


QDRANT_PAYLOAD_FIELDS = [
    "text_en",
    "passage_id",
    "query_id",
    "lang",
    "query_type",
    "source_dataset",
    "chunk_strategy",
    "chunk_id",
    "has_year",
    "has_person",
    "has_org",
    "has_location",
    "year_mentions",
    "chunk_word_count",
]


def embed_dense(
    texts: list,
    dense_model,
) -> list:

    batch_size = getattr(
        settings,
        "EMBED_BATCH_SIZE",
        32,
    )

    logger.info(
        "Encoding %s texts (dense), batch_size=%s...",
        f"{len(texts):,}",
        batch_size,
    )

    return dense_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def embed_sparse(
    texts: list,
    sparse_model,
) -> list:

    batch_size = getattr(
        settings,
        "EMBED_BATCH_SIZE",
        32,
    )

    logger.info(
        "Encoding %s texts (sparse), batch_size=%s...",
        f"{len(texts):,}",
        batch_size,
    )

    return list(
        sparse_model.embed(
            texts,
            batch_size=batch_size,
        )
    )


def build_points(
    df: pd.DataFrame,
    dense_vectors,
    sparse_vectors,
) -> list:

    points = []

    for dvec, svec, (_, row) in zip(
        dense_vectors,
        sparse_vectors,
        df.iterrows(),
    ):

        payload = {}

        for key in QDRANT_PAYLOAD_FIELDS:

            if key not in row:
                continue

            value = row[key]

            if isinstance(value, list) or pd.notna(value):
                payload[key] = value

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dvec.tolist(),
                    "sparse": SparseVector(
                        indices=svec.indices.tolist(),
                        values=svec.values.tolist(),
                    ),
                },
                payload=payload,
            )
        )

    return points


def upsert_batch(
    client: QdrantClient,
    points: list,
    batch_size: int = 256,
) -> None:

    total_batches = (
        len(points) + batch_size - 1
    ) // batch_size

    for i in range(
        0,
        len(points),
        batch_size,
    ):

        current_batch = points[
            i:i + batch_size
        ]

        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=current_batch,
        )

        logger.info(
            "Upserted batch %s/%s",
            i // batch_size + 1,
            total_batches,
        )

    logger.info(
        "Upserted %s hybrid points into '%s'",
        f"{len(points):,}",
        settings.QDRANT_COLLECTION,
    )