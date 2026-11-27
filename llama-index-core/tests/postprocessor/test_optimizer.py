"""Test optimization."""

from threading import Lock
from typing import Any, List
from unittest.mock import patch

import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.postprocessor.optimizer import SentenceEmbeddingOptimizer
from llama_index.core.schema import (
    IndexNode,
    MediaResource,
    Node,
    NodeWithScore,
    QueryBundle,
    TextNode,
)


def mock_tokenizer_fn(text: str) -> List[str]:
    """Mock tokenizer function."""
    # split by words
    return text.split(" ")


def mock_tokenizer_fn2(text: str) -> List[str]:
    """Mock tokenizer function."""
    # split by words
    return text.split(",")


def mock_get_text_embedding(text: str) -> List[float]:
    """Mock get text embedding."""
    # assume dimensions are 5
    if text == "hello":
        return [1, 0, 0, 0, 0]
    elif text == "world":
        return [0, 1, 0, 0, 0]
    elif text == "foo":
        return [0, 0, 1, 0, 0]
    elif text == "bar":
        return [0, 0, 0, 1, 0]
    elif text == "abc":
        return [0, 0, 0, 0, 1]
    else:
        raise ValueError("Invalid text for `mock_get_text_embedding`.")


def mock_get_text_embeddings(texts: List[str]) -> List[List[float]]:
    """Mock get text embeddings."""
    return [mock_get_text_embedding(text) for text in texts]


def mock_get_text_embedding_chinese(text: str) -> List[float]:
    """Mock get text embedding."""
    # assume dimensions are 5
    if text == "你":
        return [1, 0, 0, 0, 0]
    elif text == "好":
        return [0, 1, 0, 0, 0]
    elif text == "世":
        return [0, 0, 1, 0, 0]
    elif text == "界":
        return [0, 0, 0, 1, 0]
    elif text == "abc":
        return [0, 0, 0, 0, 1]
    else:
        raise ValueError("Invalid text for `mock_get_text_embedding_chinese`.", text)


def mock_get_text_embeddings_chinese(texts: List[str]) -> List[List[float]]:
    """Mock get text embeddings."""
    return [mock_get_text_embedding_chinese(text) for text in texts]


@patch.object(MockEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding)
@patch.object(
    MockEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def test_optimizer(_mock_embeds: Any, _mock_embed: Any) -> None:
    """Test optimizer."""
    optimizer = SentenceEmbeddingOptimizer(
        embed_model=MockEmbedding(embed_dim=5),
        tokenizer_fn=mock_tokenizer_fn,
        percentile_cutoff=0.5,
        context_before=0,
        context_after=0,
    )
    query = QueryBundle(query_str="hello", embedding=[1, 0, 0, 0, 0])
    orig_node = TextNode(text="hello world")
    optimized_node = optimizer.postprocess_nodes(
        [NodeWithScore(node=orig_node)], query
    )[0]
    assert optimized_node.node.get_content() == "hello"

    # test with threshold cutoff
    optimizer = SentenceEmbeddingOptimizer(
        embed_model=MockEmbedding(embed_dim=5),
        tokenizer_fn=mock_tokenizer_fn,
        threshold_cutoff=0.3,
        context_after=0,
        context_before=0,
    )
    query = QueryBundle(query_str="world", embedding=[0, 1, 0, 0, 0])
    orig_node = TextNode(text="hello world")
    optimized_node = optimizer.postprocess_nodes(
        [NodeWithScore(node=orig_node)], query
    )[0]
    assert optimized_node.node.get_content() == "world"

    # test with comma splitter
    optimizer = SentenceEmbeddingOptimizer(
        embed_model=MockEmbedding(embed_dim=5),
        tokenizer_fn=mock_tokenizer_fn2,
        threshold_cutoff=0.3,
        context_after=0,
        context_before=0,
    )
    query = QueryBundle(query_str="foo", embedding=[0, 0, 1, 0, 0])
    orig_node = TextNode(text="hello,world,foo,bar")
    optimized_node = optimizer.postprocess_nodes(
        [NodeWithScore(node=orig_node)], query
    )[0]
    assert optimized_node.node.get_content() == "foo"

    # test with further context after top sentence
    optimizer = SentenceEmbeddingOptimizer(
        embed_model=MockEmbedding(embed_dim=5),
        tokenizer_fn=mock_tokenizer_fn2,
        threshold_cutoff=0.3,
        context_after=1,
        context_before=0,
    )
    query = QueryBundle(query_str="foo", embedding=[0, 0, 1, 0, 0])
    orig_node = TextNode(text="hello,world,foo,bar")
    optimized_node = optimizer.postprocess_nodes(
        [NodeWithScore(node=orig_node)], query
    )[0]
    assert optimized_node.node.get_content() == "foo bar"

    # test with further context before and after top sentence
    optimizer = SentenceEmbeddingOptimizer(
        embed_model=MockEmbedding(embed_dim=5),
        tokenizer_fn=mock_tokenizer_fn2,
        threshold_cutoff=0.3,
        context_after=1,
        context_before=1,
    )
    query = QueryBundle(query_str="foo", embedding=[0, 0, 1, 0, 0])
    orig_node = TextNode(text="hello,world,foo,bar")
    optimized_node = optimizer.postprocess_nodes(
        [NodeWithScore(node=orig_node)], query
    )[0]
    assert optimized_node.node.get_content() == "world foo bar"


class CapitalEmbedding(MockEmbedding):
    def _get_text_embedding(self, text: str) -> List[float]:
        return [float("France" in text), float("Japan" in text)]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)


def test_optimizer_preserves_source_for_later_queries() -> None:
    optimizer = SentenceEmbeddingOptimizer(
        embed_model=CapitalEmbedding(embed_dim=2),
        tokenizer_fn=str.splitlines,
        threshold_cutoff=0.5,
        context_before=0,
        context_after=0,
    )
    source = TextNode(
        id_="capitals",
        text="Paris is in France.\nTokyo is in Japan.",
    )
    nodes = [NodeWithScore(node=source, score=0.8)]
    first = optimizer.postprocess_nodes(nodes, query_str="France")
    assert first[0].text == "Paris is in France."
    assert source.text == "Paris is in France.\nTokyo is in Japan."

    second = optimizer.postprocess_nodes(nodes, query_str="Japan")
    assert second[0].text == "Tokyo is in Japan."
    assert first[0].text == "Paris is in France."
    assert second[0].node_id == "capitals"
    assert second[0].score == 0.8


class CachedCapitalRetriever(BaseRetriever):
    def __init__(self, nodes: List[NodeWithScore], reuse_wrappers: bool) -> None:
        super().__init__()
        self.nodes = nodes
        self.reuse_wrappers = reuse_wrappers

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self.reuse_wrappers:
            return self.nodes
        return [NodeWithScore(node=n.node, score=n.score) for n in self.nodes]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("reuse_wrappers", [False, True])
async def test_optimizer_preserves_cached_query_results(
    use_async: bool, reuse_wrappers: bool
) -> None:
    source = TextNode(
        id_="capitals",
        text="Paris is in France.\nTokyo is in Japan.",
        metadata={"source": "atlas"},
        excluded_llm_metadata_keys=["source"],
    )
    cached = [NodeWithScore(node=source, score=0.8)]
    retriever = CachedCapitalRetriever(cached, reuse_wrappers=reuse_wrappers)
    optimizer = SentenceEmbeddingOptimizer(
        embed_model=CapitalEmbedding(embed_dim=2),
        tokenizer_fn=str.splitlines,
        threshold_cutoff=0.5,
        context_before=0,
        context_after=0,
    )
    engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=MockLLM(),
        response_mode="no_text",
        node_postprocessors=[optimizer],
    )
    if use_async:
        first = await engine.aquery("France")
        second = await engine.aquery("Japan")
    else:
        first = engine.query("France")
        second = engine.query("Japan")

    assert first.source_nodes[0].text == "Paris is in France."
    assert second.source_nodes[0].text == "Tokyo is in Japan."
    assert source.text == "Paris is in France.\nTokyo is in Japan."
    assert cached[0].score == 0.8
    assert second.source_nodes[0].node_id == "capitals"
    assert second.source_nodes[0].metadata == {"source": "atlas"}
    assert second.source_nodes[0].score == 0.8


def test_optimizer_failure_does_not_shorten_previous_nodes() -> None:
    optimizer = SentenceEmbeddingOptimizer(
        embed_model=CapitalEmbedding(embed_dim=2),
        tokenizer_fn=str.splitlines,
        threshold_cutoff=0.5,
        context_before=0,
        context_after=0,
    )
    source = TextNode(text="Paris is in France.\nTokyo is in Japan.")
    nodes = [
        NodeWithScore(node=source),
        NodeWithScore(node=TextNode(text="Tokyo is in Japan.")),
    ]
    with pytest.raises(ValueError, match="Optimizer returned zero sentences"):
        optimizer.postprocess_nodes(nodes, query_str="France")
    assert source.text == "Paris is in France.\nTokyo is in Japan."

    assert optimizer.postprocess_nodes(nodes) is nodes
    assert optimizer.postprocess_nodes([], query_str="France") == []


@pytest.mark.parametrize("node_type", ["index", "node"])
def test_optimizer_preserves_node_resources(node_type: str) -> None:
    text = "Paris is in France.\nTokyo is in Japan."
    lock = Lock()
    resource = MediaResource(text=text)
    source = (
        IndexNode(text=text, index_id="reference", obj=lock)
        if node_type == "index"
        else Node(text_resource=resource)
    )
    optimizer = SentenceEmbeddingOptimizer(
        embed_model=CapitalEmbedding(embed_dim=2),
        tokenizer_fn=str.splitlines,
        threshold_cutoff=0.5,
        context_before=0,
        context_after=0,
    )
    nodes = [NodeWithScore(node=source)]
    first = optimizer.postprocess_nodes(nodes, query_str="France")
    second = optimizer.postprocess_nodes(nodes, query_str="Japan")
    assert first[0].get_content() == "Paris is in France."
    assert second[0].get_content() == "Tokyo is in Japan."
    assert source.get_content() == text
    if isinstance(source, IndexNode):
        assert isinstance(first[0].node, IndexNode)
        assert isinstance(second[0].node, IndexNode)
        assert first[0].node.obj is lock
        assert second[0].node.obj is lock
    else:
        assert source.text_resource is resource
        assert resource.text == text
