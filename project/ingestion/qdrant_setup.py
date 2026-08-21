"""
ingestion/qdrant_setup.py

Qdrant collection creation + payload index setup. Split out from
embed_and_upsert_hybrid (notebook, post-Cell 19) -- this file owns
only collection/schema setup; embedding + upserting points lives in
ingestion/upsert.py.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PayloadSchemaType

from project.config.settings import settings
from project.logging_system.logger import get_logger

logger = get_logger(__name__)

# Same 9 payload index fields as the notebook's index_fields list.
PAYLOAD_INDEX_FIELDS = [
    ("lang", PayloadSchemaType.KEYWORD),
    ("query_type", PayloadSchemaType.KEYWORD),
    ("chunk_strategy", PayloadSchemaType.KEYWORD),
    ("source_dataset", PayloadSchemaType.KEYWORD),
    ("has_year", PayloadSchemaType.BOOL),
    ("has_person", PayloadSchemaType.BOOL),
    ("has_org", PayloadSchemaType.BOOL),
    ("has_location", PayloadSchemaType.BOOL),
    ("year_mentions", PayloadSchemaType.INTEGER),
]


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL)


def ensure_collection(client: QdrantClient, recreate: bool = True) -> None:
    """Create (or recreate) the hybrid dense+sparse named-vector collection."""
    if recreate and client.collection_exists(settings.QDRANT_COLLECTION):
        client.delete_collection(collection_name=settings.QDRANT_COLLECTION)

    if recreate or not client.collection_exists(settings.QDRANT_COLLECTION):
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config={"dense": VectorParams(size=settings.EMBED_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        logger.info("Created collection '%s'", settings.QDRANT_COLLECTION)


def ensure_payload_indexes(client: QdrantClient) -> None:
    for field, schema in PAYLOAD_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION, field_name=field, field_schema=schema
        )
    logger.info("Created %s payload indexes.", len(PAYLOAD_INDEX_FIELDS))
