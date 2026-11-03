from unittest.mock import MagicMock, patch

import pytest

from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.redis.schema import RedisVectorStoreSchema
from redisvl.schema import IndexSchema


@pytest.mark.parametrize(
    ("field_name", "field_type"),
    [("id", "tag"), ("doc_id", "tag"), ("text", "text"), ("vector", "vector")],
)
@pytest.mark.parametrize("invalid_field", ["missing", "wrong_type"])
def test_invalid_required_schema_field(field_name, field_type, invalid_field):
    schema_dict = RedisVectorStoreSchema().to_dict()
    fields = [field for field in schema_dict["fields"] if field["name"] != field_name]
    if invalid_field == "wrong_type":
        fields.append(
            {"name": field_name, "type": "text" if field_type == "tag" else "tag"}
        )
    schema_dict["fields"] = fields
    schema = IndexSchema.from_dict(schema_dict)

    with patch("llama_index.vector_stores.redis.base.SearchIndex") as search_index:
        with pytest.raises(
            ValueError,
            match=f"Required field {field_name} must be present in the index and of type {field_type}",
        ):
            RedisVectorStore(schema=schema, redis_client_async=MagicMock())
        search_index.assert_not_called()


@pytest.mark.parametrize("extra_field", [False, True])
def test_valid_schema(extra_field):
    schema = RedisVectorStoreSchema()
    if extra_field:
        schema.add_fields([{"name": "category", "type": "tag"}])
    client = MagicMock()
    with (
        patch("llama_index.vector_stores.redis.base.SearchIndex") as search_index,
        patch("llama_index.vector_stores.redis.base.AsyncSearchIndex") as async_index,
    ):
        RedisVectorStore(schema=schema, redis_client_async=client)
        search_index.assert_called_once_with(
            schema=schema, redis_client=None, redis_url=None
        )
        async_index.assert_called_once_with(schema=schema, redis_client=client)
