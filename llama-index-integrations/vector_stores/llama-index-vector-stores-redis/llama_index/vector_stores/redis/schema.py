from pydantic.v1 import Field
from typing import Dict
from redisvl.schema import (
    IndexSchema,
    IndexInfo,
    StorageType
)
from redisvl.schema.fields import (
    BaseField,
    FieldFactory
)

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
    def __init__(self, **data):
        index = RedisIndexInfo()
        fields: Dict[str, BaseField] = {
            "id": FieldFactory.create_field("tag", "id", {"sortable": False}),
            "doc_id": FieldFactory.create_field("tag", "doc_id", {"sortable": False}),
            "text": FieldFactory.create_field("text", "text", {"weight": 1.0}),
        }
        super().__init__(index=index, fields=fields)
