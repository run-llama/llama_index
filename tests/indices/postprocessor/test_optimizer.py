"""Test optimization."""

import pytest
from typing import Any, List
from unittest.mock import patch

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.postprocessor.optimizer import SentenceEmbeddingOptimizer
from llama_index.schema import TextNode, NodeWithScore
from llama_index.utils import (
    get_transformer_tokenizer_fn,
)

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None  # type: ignore


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


@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def test_optimizer(_mock_embeds: Any, _mock_embed: Any) -> None:
    """Test optimizer."""
    optimizer = SentenceEmbeddingOptimizer(
        tokenizer_fn=mock_tokenizer_fn, percentile_cutoff=0.5
    )
    query = QueryBundle(query_str="hello", embedding=[1, 0, 0, 0, 0])
    orig_node = TextNode(text="hello world")
    optimized_node = optimizer.postprocess_nodes(
        [NodeWithScore(node=orig_node)], query
    )[0]
    assert optimized_node.node.get_content() == "hello"

    # test with threshold cutoff
    optimizer = SentenceEmbeddingOptimizer(
        tokenizer_fn=mock_tokenizer_fn, threshold_cutoff=0.3
    )
    query = QueryBundle(query_str="world", embedding=[0, 1, 0, 0, 0])
    orig_node = TextNode(text="hello world")
    optimized_node = optimizer.postprocess_nodes(
        [NodeWithScore(node=orig_node)], query
    )[0]
    assert optimized_node.node.get_content() == "world"

    # test with comma splitter
    optimizer = SentenceEmbeddingOptimizer(
        tokenizer_fn=mock_tokenizer_fn2, threshold_cutoff=0.3
    )
    query = QueryBundle(query_str="foo", embedding=[0, 0, 1, 0, 0])
    orig_node = TextNode(text="hello,world,foo,bar")
    optimized_node = optimizer.postprocess_nodes(
        [NodeWithScore(node=orig_node)], query
    )[0]
    assert optimized_node.node.get_content() == "foo"


@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embeddings_chinese
)
@patch.object(
    OpenAIEmbedding,
    "_get_text_embeddings",
    side_effect=mock_get_text_embeddings_chinese,
)
@pytest.mark.skipif(AutoTokenizer is None, reason="transformers not installed")
def test_optimizer_chinese(_mock_embeds: Any, _mock_embed: Any) -> None:
    """Test optimizer."""
    optimizer = SentenceEmbeddingOptimizer(
        tokenizer_fn=get_transformer_tokenizer_fn("GanymedeNil/text2vec-large-chinese"), 
        percentile_cutoff=0.5
    )
    query = QueryBundle(query_str="你好 世界", embedding=[1, 0, 0, 0, 0])
    orig_node = TextNode(text="你好 世界")
    optimized_node = optimizer.postprocess_nodes(
        [NodeWithScore(node=orig_node)], query
    )[0]
    assert len(optimized_node.node.get_content()) < len(
        TextNode(text="你好 世界").get_content()
    )
