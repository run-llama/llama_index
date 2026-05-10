"""Tests for load_embed_model in loading.py"""

import pytest
from llama_index.core.embeddings.loading import (
    RECOGNIZED_EMBEDDINGS,
    load_embed_model,
)


def test_load_embed_model_missing_class_name():
    """Raises ValueError when class_name key is absent."""
    with pytest.raises(ValueError, match="Embedding loading requires a class_name"):
        load_embed_model({})


def test_load_embed_model_invalid_name_shows_available():
    """Error message for bad name must list available embeddings."""
    with pytest.raises(ValueError) as exc_info:
        load_embed_model({"class_name": "NonExistentEmbedding"})

    error_msg = str(exc_info.value)
    assert "Invalid Embedding name: NonExistentEmbedding" in error_msg
    assert "Available embeddings:" in error_msg
    # Every key in the live registry must appear in the message
    for key in sorted(RECOGNIZED_EMBEDDINGS):
        assert key in error_msg


def test_load_embed_model_valid_mock():
    """MockEmbedding (always present) loads without error."""
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    model = load_embed_model(MockEmbedding(embed_dim=8).to_dict())
    assert isinstance(model, MockEmbedding)
