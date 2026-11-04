import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from llama_index.core.schema import TextNode
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.redis.schema import VECTOR_FIELD_NAME

NODE_ID = "e7b95ae7-6369-404d-8287-1f4504121563"
PREFIX = "semantic_cache_doc"
SEPARATOR = ":"


def _fake_index(keys):
    index = MagicMock()
    index.prefix = PREFIX
    index.key_separator = SEPARATOR
    index.name = "test_index"
    index.schema = SimpleNamespace(
        fields={VECTOR_FIELD_NAME: SimpleNamespace(attrs=SimpleNamespace(dims=3))}
    )
    index.load.return_value = keys
    return index


def _store(index, async_index):
    store = RedisVectorStore.model_construct(
        flat_metadata=False, created_async_index=True
    )
    store._index = index
    store._async_index = async_index
    store._overwrite = False
    return store


def _node():
    return TextNode(text="test", id_=NODE_ID, embedding=[0.0, 0.0, 0.0])


def test_add_preserves_leading_hex_characters_in_node_id():
    key = f"{PREFIX}{SEPARATOR}{NODE_ID}"
    store = _store(_fake_index([key]), MagicMock())

    assert store.add([_node()]) == [NODE_ID]


def test_async_add_preserves_leading_hex_characters_in_node_id():
    key = f"{PREFIX}{SEPARATOR}{NODE_ID}"
    index = _fake_index([key])
    async_index = _fake_index([key])
    async_index.load = AsyncMock(return_value=[key])
    store = _store(index, async_index)

    assert asyncio.run(store.async_add([_node()])) == [NODE_ID]
