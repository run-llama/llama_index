"""Test optimization utils, which is other functionalities of tokenize."""

from typing import Any, List
from unittest.mock import patch

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.query.schema import QueryBundle
from llama_index.optimization.optimizer import SentenceEmbeddingOptimizer
from llama_index.optimization.utils import (
    get_large_chinese_tokenizer_fn,
    get_transformer_tokenizer_fin,
)


def mock_get_text_embedding(text: str) -> List[float]:
    """Mock get text embedding."""
    # assume dimensions are 5
    if text == "你" or text == "▁Hello":
        return [1, 0, 0, 0, 0]
    elif text == "好" or text == "▁World":
        return [0, 1, 0, 0, 0]
    elif text == "世":
        return [0, 0, 1, 0, 0]
    elif text == "界":
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
        tokenizer_fn=get_large_chinese_tokenizer_fn(), percentile_cutoff=0.5
    )
    query = QueryBundle(query_str="你好 世界", embedding=[1, 0, 0, 0, 0])
    orig_txt = "你好 世界"
    optimized_txt = optimizer.optimize(query, orig_txt)
    assert len(optimized_txt) < len(orig_txt)

    optimizer = SentenceEmbeddingOptimizer(
        tokenizer_fn=get_transformer_tokenizer_fin("fxmarty/tiny-llama-fast-tokenizer"),
        percentile_cutoff=0.5,
    )
    query = QueryBundle(query_str="Hello World", embedding=[1, 0, 0, 0, 0])
    orig_txt = "Hello World"
    optimized_txt = optimizer.optimize(query, orig_txt)
    print(optimized_txt)
    assert len(optimized_txt) < len(orig_txt)
