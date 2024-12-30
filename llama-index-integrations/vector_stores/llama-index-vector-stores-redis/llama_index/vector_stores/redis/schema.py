from typing import Any, Dict, List
from redisvl.schema import IndexSchema, IndexInfo, StorageType

# required llama index fields
NODE_ID_FIELD_NAME: str = "id"
DOC_ID_FIELD_NAME: str = "doc_id"
TEXT_FIELD_NAME: str = "text"
NODE_CONTENT_FIELD_NAME: str = "_node_content"
VECTOR_FIELD_NAME: str = "vector"


class RedisIndexInfo(IndexInfo):
    """The default Redis Vector Store Index Info."""

    name: str = "llama_index"
    """The unique name of the index."""
    prefix: str = "llama_index/vector"
    """The prefix used for Redis keys associated with this index."""
    key_separator: str = "_"
    """The separator character used in designing Redis keys."""
    storage_type: StorageType = StorageType.HASH
    """The storage type used in Redis (e.g., 'hash' or 'json')."""


class RedisVectorStoreSchema(IndexSchema):
    """The default Redis Vector Store Schema."""

    def __init__(self, **data) -> None:
        index = RedisIndexInfo()
        fields: List[Dict[str, Any]] = [
            {"type": "tag", "name": NODE_ID_FIELD_NAME, "attrs": {"sortable": False}},
            {"type": "tag", "name": DOC_ID_FIELD_NAME, "attrs": {"sortable": False}},
            {"type": "text", "name": TEXT_FIELD_NAME, "attrs": {"weight": 1.0}},
            {
                "type": "vector",
                "name": VECTOR_FIELD_NAME,
                "attrs": {
                    "dims": 1536,
                    "algorithm": "flat",
                    "distance_metric": "cosine",
                },
            },
        ]
        super().__init__(index=index.__dict__, fields=fields)
