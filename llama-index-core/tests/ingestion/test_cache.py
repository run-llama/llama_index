from typing import Any, Dict, List, Optional

import pytest
from llama_index.core.ingestion import IngestionCache
from llama_index.core.ingestion.pipeline import (
    arun_transformations,
    get_transformation_hash,
)
from llama_index.core.schema import BaseNode, TextNode, TransformComponent
from llama_index.core.storage.kvstore.types import DEFAULT_COLLECTION, BaseKVStore


class DummyTransform(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        for node in nodes:
            node.set_content(node.get_content() + "\nTESTTEST")
        return nodes


class AsyncOnlyKVStore(BaseKVStore):
    """A KVStore whose sync methods blow up, so a caller that used them would fail."""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, dict]] = {}

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        raise AssertionError("sync put() must not be called from the async path")

    async def aput(
        self, key: str, val: dict, collection: str = DEFAULT_COLLECTION
    ) -> None:
        self._data.setdefault(collection, {})[key] = val

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        raise AssertionError("sync get() must not be called from the async path")

    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        return self._data.get(collection, {}).get(key)

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        raise AssertionError("sync get_all() must not be called from the async path")

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        return dict(self._data.get(collection, {}))

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        raise AssertionError("sync delete() must not be called from the async path")

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        return self._data.get(collection, {}).pop(key, None) is not None


def test_cache() -> None:
    cache = IngestionCache()
    transformation = DummyTransform()

    node = TextNode(text="dummy")
    hash = get_transformation_hash([node], transformation)

    new_nodes = transformation([node])
    cache.put(hash, new_nodes)

    cache_hit = cache.get(hash)
    assert cache_hit is not None
    assert cache_hit[0].get_content() == new_nodes[0].get_content()

    new_hash = get_transformation_hash(new_nodes, transformation)
    assert cache.get(new_hash) is None


def test_cache_clear() -> None:
    cache = IngestionCache()
    transformation = DummyTransform()

    node = TextNode(text="dummy")
    hash = get_transformation_hash([node], transformation)

    new_nodes = transformation([node])
    cache.put(hash, new_nodes)

    cache_hit = cache.get(hash)
    assert cache_hit is not None

    cache.clear()
    assert cache.get(hash) is None


@pytest.mark.asyncio
async def test_cache_async_methods_use_async_kvstore_calls() -> None:
    """IngestionCache's async methods must call the KVStore's async methods."""
    cache = IngestionCache(cache=AsyncOnlyKVStore())
    transformation = DummyTransform()

    node = TextNode(text="dummy")
    hash = get_transformation_hash([node], transformation)

    new_nodes = transformation([node])
    await cache.aput(hash, new_nodes)

    cache_hit = await cache.aget(hash)
    assert cache_hit is not None
    assert cache_hit[0].get_content() == new_nodes[0].get_content()

    await cache.aclear()
    assert await cache.aget(hash) is None


@pytest.mark.asyncio
async def test_arun_transformations_uses_async_cache_path() -> None:
    """
    Regression test for the async ingestion pipeline blocking the event loop.

    `arun_transformations` previously called `IngestionCache.get`/`.put`, which
    are synchronous. With a cache backend whose sync methods raise (simulating
    a remote cache that only implements the async KVStore API meaningfully),
    the async pipeline must still succeed by going through `aget`/`aput`.
    """
    cache = IngestionCache(cache=AsyncOnlyKVStore())
    transformation = DummyTransform()
    node = TextNode(text="dummy")

    nodes = await arun_transformations([node], [transformation], cache=cache)
    assert nodes[0].get_content().endswith("TESTTEST")

    # second run should hit the cache instead of re-running the transform
    cached_nodes = await arun_transformations([node], [transformation], cache=cache)
    assert cached_nodes[0].get_content() == nodes[0].get_content()
