from typing import Any, List

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


def test_transformation_hash_distinguishes_node_boundaries() -> None:
    """
    Distinct node splits of the same text must not share a transformation-cache entry.

    `get_transformation_hash` joined node contents without a separator, so `["ab", "c"]` and
    `["a", "bc"]` both joined to `"abc"` and hashed identically: one input could be served the
    other's cached nodes, which carry their own `id_`/`ref_doc_id`.
    """
    from llama_index.core.ingestion.pipeline import get_transformation_hash
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import TextNode

    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0)
    two_and_one = [TextNode(text="ab", id_="a1"), TextNode(text="c", id_="a2")]
    one_and_two = [TextNode(text="a", id_="b1"), TextNode(text="bc", id_="b2")]

    assert "".join(node.text for node in two_and_one) == "".join(
        node.text for node in one_and_two
    )
    assert get_transformation_hash(two_and_one, splitter) != get_transformation_hash(
        one_and_two, splitter
    )
