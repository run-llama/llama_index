from typing import List
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from unittest.mock import patch

import pytest


def mock_get_text_embedding(text: str) -> List[float]:
    """Mock get text embedding."""
    # assume dimensions are 5
    if text == "Jerry likes juice.":
        return [1, 1, 0, 0, 0]
    elif text == "Bob likes burgers.":
        return [0, 1, 0, 1, 0]
    elif text == "Alice likes apples.":
        return [0, 0, 1, 0, 0]
    elif text == "What does Jerry like?":
        return [1, 1, 0, 0, 1]
    elif (
        text == "Jerry likes juice. That's nice."
    ):  # vector memory batches conversation turns starting with user
        return [1, 1, 0, 0, 1]
    else:
        raise ValueError("Invalid text for `mock_get_text_embedding`.")


def mock_get_text_embeddings(texts: List[str]) -> List[List[float]]:
    """Mock get text embeddings."""
    return [mock_get_text_embedding(text) for text in texts]


@pytest.fixture()
@patch.object(MockEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding)
@patch.object(
    MockEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def mock_embed_model() -> MockEmbedding:
    return MockEmbedding(embed_dim=5)
