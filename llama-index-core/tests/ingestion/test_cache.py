from typing import Any, List

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.ingestion import IngestionCache
from llama_index.core.ingestion.pipeline import (
    get_transformation_hash,
    run_transformations,
)
from llama_index.core.schema import (
    BaseNode,
    NodeRelationship,
    TextNode,
    TransformComponent,
)


class DummyTransform(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        for node in nodes:
            node.set_content(node.get_content() + "\nTESTTEST")
        return nodes


class RandomIdSplitTransform(TransformComponent):
    """
    Mimics a real splitter using the default `id_func`: every call produces
    brand-new node ids for its output, even when the input is unchanged, but
    the output nodes carry a SOURCE relationship back to the (stable) input
    node, exactly like `build_nodes_from_splits` does.
    """

    @classmethod
    def class_name(cls) -> str:
        return "random_id_split_transform"

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        chunks = []
        for node in nodes:
            chunk = TextNode(text=node.get_content())
            chunk.relationships[NodeRelationship.SOURCE] = node.as_related_node_info()
            chunks.append(chunk)
        return chunks


class CountingTransform(TransformComponent):
    # A private attribute so the running count isn't part of `to_dict()` and
    # doesn't itself perturb the transformation hash between calls.
    _call_count: int = PrivateAttr(default=0)

    @classmethod
    def class_name(cls) -> str:
        return "counting_transform"

    @property
    def call_count(self) -> int:
        return self._call_count

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        self._call_count += 1
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


def test_get_transformation_hash_is_injective() -> None:
    transformation = DummyTransform()

    # Splitting the same combined text at a different node boundary must not
    # collide: ["ab", "c"] and ["a", "bc"] previously hashed identically because
    # node contents were concatenated with no separator.
    nodes_ab_c = [TextNode(id_="1", text="ab"), TextNode(id_="2", text="c")]
    nodes_a_bc = [TextNode(id_="3", text="a"), TextNode(id_="4", text="bc")]
    assert get_transformation_hash(
        nodes_ab_c, transformation
    ) != get_transformation_hash(nodes_a_bc, transformation)

    # Two distinct documents with identical content must not hash the same,
    # since the cached value carries node identity (id_, ref_doc_id) that the
    # key must also depend on.
    node_a = TextNode(id_="doc_a", text="same content")
    node_b = TextNode(id_="doc_b", text="same content")
    assert get_transformation_hash([node_a], transformation) != get_transformation_hash(
        [node_b], transformation
    )


def test_downstream_cache_survives_upstream_id_regeneration() -> None:
    """
    A splitter's output nodes get fresh random ids on every run (the default
    `id_func`), even when the input document is unchanged, so the hash used to
    key a *later* stage's cache entry must not depend on those transient ids.
    Otherwise, evicting only the upstream stage's cache entry would silently
    cascade into a downstream cache miss too - forcing an expensive stage
    (e.g. embedding) to redundantly re-run even though the document content
    never changed.
    """
    cache = IngestionCache()
    split = RandomIdSplitTransform()
    embed = CountingTransform()
    doc = TextNode(id_="stable-doc-id", text="unchanged content")

    run_transformations([doc], [split, embed], cache=cache)
    assert embed.call_count == 1

    # Evict only the splitter's cache entry, forcing it to re-run and produce
    # chunk nodes with brand-new random ids on the next call.
    split_hash = get_transformation_hash([doc], split)
    cache.cache.delete(split_hash, collection=cache.collection)

    run_transformations([doc], [split, embed], cache=cache)
    assert embed.call_count == 1  # still cached: content never changed


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
