from typing import Any, List

from llama_index.core.ingestion import IngestionCache
from llama_index.core.ingestion.pipeline import get_transformation_hash
from llama_index.core.schema import BaseNode, TextNode, TransformComponent


class DummyTransform(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        for node in nodes:
            node.set_content(node.get_content() + "\nTESTTEST")
        return nodes


class OtherDummyTransform(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        return nodes


class NamedTransform(DummyTransform):
    @classmethod
    def class_name(cls) -> str:
        return "NamedTransform"


class RenamedTransform(DummyTransform):
    @classmethod
    def class_name(cls) -> str:
        return "NamedTransform"


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


def test_transformation_hash_differs_between_unnamed_transformations() -> None:
    # neither class overrides class_name(), so they serialize identically
    assert DummyTransform().to_dict() == OtherDummyTransform().to_dict()

    nodes = [TextNode(text="dummy")]
    hash = get_transformation_hash(nodes, DummyTransform())

    assert hash != get_transformation_hash(nodes, OtherDummyTransform())
    assert hash == get_transformation_hash(nodes, DummyTransform())


def test_transformation_hash_follows_declared_class_name() -> None:
    # a declared class_name() stays the identity, so renaming or moving such a
    # class does not invalidate its cache entries
    nodes = [TextNode(text="dummy")]

    assert get_transformation_hash(nodes, NamedTransform()) == get_transformation_hash(
        nodes, RenamedTransform()
    )
