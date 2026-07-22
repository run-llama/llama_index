from typing import Any, List

import pytest

from llama_index.core.ingestion import IngestionCache
from llama_index.core.ingestion.pipeline import get_transformation_hash
from llama_index.core.schema import BaseNode, TextNode, TransformComponent


class DummyTransform(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        for node in nodes:
            node.set_content(node.get_content() + "\nTESTTEST")
        return nodes


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
async def test_async_cache() -> None:
    cache = IngestionCache()
    transformation = DummyTransform()

    node = TextNode(text="dummy")
    hash = get_transformation_hash([node], transformation)

    new_nodes = transformation([node])
    await cache.aput(hash, new_nodes)

    cache_hit = await cache.aget(hash)
    assert cache_hit is not None
    assert cache_hit[0].get_content() == new_nodes[0].get_content()

    new_hash = get_transformation_hash(new_nodes, transformation)
    new_hash_cache_hit = await cache.aget(new_hash)
    assert new_hash_cache_hit is None


@pytest.mark.asyncio
async def test__async_cache_clear() -> None:
    cache = IngestionCache()
    transformation = DummyTransform()

    node1 = TextNode(text="dummy1")
    node2 = TextNode(text="dummy2")
    hash = get_transformation_hash([node1, node2], transformation)

    new_nodes = transformation([node1, node2])
    cache.put(hash, new_nodes)

    cache_hit = await cache.aget(hash)
    assert cache_hit is not None

    await cache.aclear()
    new_cache_hit = await cache.aget(hash)
    assert new_cache_hit is None
