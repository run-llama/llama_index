"""Embeddings."""
from typing import Any, List
from unittest.mock import patch
from llama_index.embeddings.base import mean_agg

from llama_index.embeddings.openai import OpenAIEmbedding


def mock_get_text_embedding(text: str) -> List[float]:
    """Mock get text embedding."""
    # assume dimensions are 5
    if text == "Hello world.":
        return [1, 0, 0, 0, 0]
    elif text == "This is a test.":
        return [0, 1, 0, 0, 0]
    elif text == "This is another test.":
        return [0, 0, 1, 0, 0]
    elif text == "This is a test v2.":
        return [0, 0, 0, 1, 0]
    elif text == "This is a test v3.":
        return [0, 0, 0, 0, 1]
    elif text == "This is bar test.":
        return [0, 0, 1, 0, 0]
    elif text == "Hello world backup.":
        # this is used when "Hello world." is deleted.
        return [1, 0, 0, 0, 0]
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
def test_get_queued_text_embeddings(
    _mock_get_text_embeddings: Any, _mock_get_text_embedding: Any
) -> None:
    """Test get queued text embeddings."""
    embed_model = OpenAIEmbedding(embed_batch_size=8)
    for i in range(8):
        embed_model.queue_text_for_embedding(f"id:{i}", "Hello world.")
    for i in range(8):
        embed_model.queue_text_for_embedding(f"id:{i}", "This is a test.")
    for i in range(4):
        embed_model.queue_text_for_embedding(f"id:{i}", "This is another test.")
    for i in range(4):
        embed_model.queue_text_for_embedding(f"id:{i}", "This is a test v2.")

    result_ids, result_embeddings = embed_model.get_queued_text_embeddings()
    for i in range(8):
        assert result_ids[i] == f"id:{i}"
        assert result_embeddings[i] == [1, 0, 0, 0, 0]
    for i in range(8, 16):
        assert result_ids[i] == f"id:{i-8}"
        assert result_embeddings[i] == [0, 1, 0, 0, 0]
    for i in range(16, 20):
        assert result_ids[i] == f"id:{i-16}"
        assert result_embeddings[i] == [0, 0, 1, 0, 0]
    for i in range(20, 24):
        assert result_ids[i] == f"id:{i-20}"
        assert result_embeddings[i] == [0, 0, 0, 1, 0]


def test_embedding_similarity() -> None:
    """Test embedding similarity."""
    embed_model = OpenAIEmbedding()
    text_embedding = [3.0, 4.0, 0.0]
    query_embedding = [0.0, 1.0, 0.0]
    cosine = embed_model.similarity(query_embedding, text_embedding)
    assert cosine == 0.8


def test_mean_agg() -> None:
    """Test mean aggregation for embeddings."""
    embedding_0 = [3.0, 4.0, 0.0]
    embedding_1 = [0.0, 1.0, 0.0]
    output = mean_agg([embedding_0, embedding_1])
    assert output == [1.5, 2.5, 0.0]
