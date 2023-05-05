"""Test optimization."""

from typing import Any, List
from unittest.mock import patch

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.query.schema import QueryBundle
from llama_index.optimization.optimizer import SentenceEmbeddingOptimizer


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
    orig_txt = "hello world"
    optimized_txt = optimizer.optimize(query, orig_txt)
    assert optimized_txt == "hello"

    # test with threshold cutoff
    optimizer = SentenceEmbeddingOptimizer(
        tokenizer_fn=mock_tokenizer_fn, threshold_cutoff=0.3
    )
    query = QueryBundle(query_str="world", embedding=[0, 1, 0, 0, 0])
    orig_txt = "hello world foo bar"
    optimized_txt = optimizer.optimize(query, orig_txt)
    assert optimized_txt == "world"

    # test with comma splitter
    optimizer = SentenceEmbeddingOptimizer(
        tokenizer_fn=mock_tokenizer_fn2, threshold_cutoff=0.3
    )
    query = QueryBundle(query_str="foo", embedding=[0, 0, 1, 0, 0])
    orig_txt = "hello,world,foo,bar"
    optimized_txt = optimizer.optimize(query, orig_txt)
    assert optimized_txt == "foo"
