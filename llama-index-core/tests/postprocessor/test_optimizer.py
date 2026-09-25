"""Test optimization."""

import asyncio
from typing import Any, List
from unittest.mock import patch

import pytest

from llama_index.core.async_utils import DEFAULT_NUM_WORKERS

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.postprocessor.optimizer import SentenceEmbeddingOptimizer
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


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


async def amock_get_text_embeddings(texts: List[str]) -> List[List[float]]:
    """Mock async get text embeddings."""
    return mock_get_text_embeddings(texts)


@patch.object(
    MockEmbedding, "_aget_text_embeddings", side_effect=amock_get_text_embeddings
)
@pytest.mark.asyncio
async def test_aoptimizer(_mock_aembeds: Any) -> None:
    """Async optimization must match the sync result."""
    optimizer = SentenceEmbeddingOptimizer(
        embed_model=MockEmbedding(embed_dim=5),
        tokenizer_fn=mock_tokenizer_fn,
        percentile_cutoff=0.5,
        context_before=0,
        context_after=0,
    )
    query = QueryBundle(query_str="hello", embedding=[1, 0, 0, 0, 0])
    optimized = await optimizer.apostprocess_nodes(
        [NodeWithScore(node=TextNode(text="hello world"))], query
    )
    assert optimized[0].node.get_content() == "hello"

    # test with threshold cutoff
    optimizer = SentenceEmbeddingOptimizer(
        embed_model=MockEmbedding(embed_dim=5),
        tokenizer_fn=mock_tokenizer_fn,
        threshold_cutoff=0.3,
        context_after=0,
        context_before=0,
    )
    query = QueryBundle(query_str="world", embedding=[0, 1, 0, 0, 0])
    optimized = await optimizer.apostprocess_nodes(
        [NodeWithScore(node=TextNode(text="hello world"))], query
    )
    assert optimized[0].node.get_content() == "world"

    # test with context before and after the top sentence
    optimizer = SentenceEmbeddingOptimizer(
        embed_model=MockEmbedding(embed_dim=5),
        tokenizer_fn=mock_tokenizer_fn2,
        threshold_cutoff=0.3,
        context_after=1,
        context_before=1,
    )
    query = QueryBundle(query_str="foo", embedding=[0, 0, 1, 0, 0])
    optimized = await optimizer.apostprocess_nodes(
        [NodeWithScore(node=TextNode(text="hello,world,foo,bar"))], query
    )
    assert optimized[0].node.get_content() == "world foo bar"


@patch.object(
    MockEmbedding, "_aget_text_embeddings", side_effect=amock_get_text_embeddings
)
@pytest.mark.asyncio
async def test_aoptimizer_multiple_nodes(_mock_aembeds: Any) -> None:
    """Every node is optimized, and the input list is returned in order."""
    optimizer = SentenceEmbeddingOptimizer(
        embed_model=MockEmbedding(embed_dim=5),
        tokenizer_fn=mock_tokenizer_fn2,
        threshold_cutoff=0.3,
        context_after=0,
        context_before=0,
    )
    query = QueryBundle(query_str="foo", embedding=[0, 0, 1, 0, 0])
    nodes = [
        NodeWithScore(node=TextNode(text="hello,world,foo"), score=0.9),
        NodeWithScore(node=TextNode(text="bar,foo"), score=0.5),
        NodeWithScore(node=TextNode(text="foo,abc"), score=0.1),
    ]
    optimized = await optimizer.apostprocess_nodes(nodes, query)

    assert [n.node.get_content() for n in optimized] == ["foo", "foo", "foo"]
    assert [n.score for n in optimized] == [0.9, 0.5, 0.1]


@pytest.mark.asyncio
async def test_aoptimizer_runs_nodes_concurrently() -> None:
    """Every node should be embedded in parallel, not one at a time."""
    # 3 nodes, deliberately BELOW async_utils.DEFAULT_NUM_WORKERS (4) so the
    # assertion measures this method's dispatch, not run_jobs' semaphore ceiling.
    nodes = [
        NodeWithScore(node=TextNode(text="hello,world,foo")),
        NodeWithScore(node=TextNode(text="bar,foo")),
        NodeWithScore(node=TextNode(text="foo,abc")),
    ]
    assert len(nodes) < DEFAULT_NUM_WORKERS

    in_flight = 0
    max_in_flight = 0

    async def tracking_embeds(texts: List[str]) -> List[List[float]]:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            # yield control so a sequential implementation cannot overlap
            await asyncio.sleep(0)
            return mock_get_text_embeddings(texts)
        finally:
            in_flight -= 1

    optimizer = SentenceEmbeddingOptimizer(
        embed_model=MockEmbedding(embed_dim=5),
        tokenizer_fn=mock_tokenizer_fn2,
        threshold_cutoff=0.3,
        context_after=0,
        context_before=0,
    )
    query = QueryBundle(query_str="foo", embedding=[0, 0, 1, 0, 0])

    with patch.object(
        MockEmbedding, "_aget_text_embeddings", side_effect=tracking_embeds
    ):
        await optimizer.apostprocess_nodes(nodes, query)

    assert max_in_flight == len(nodes), (
        f"expected all {len(nodes)} nodes in flight concurrently, "
        f"saw at most {max_in_flight}"
    )
