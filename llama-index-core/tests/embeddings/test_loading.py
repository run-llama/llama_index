"""Tests for load_embed_model in loading.py."""

import pytest

from llama_index.core.embeddings.loading import (
    RECOGNIZED_EMBEDDINGS,
    load_embed_model,
)
from llama_index.core.embeddings.mock_embed_model import MockEmbedding


def test_load_embed_model_missing_class_name() -> None:
    """Raises ValueError when class_name key is absent."""
    with pytest.raises(ValueError, match="Embedding loading requires a class_name"):
        load_embed_model({})


def test_load_embed_model_invalid_name_shows_available() -> None:
    """Error message for an unrecognized name must list all available embeddings."""
    with pytest.raises(ValueError) as exc_info:
        load_embed_model({"class_name": "NonExistentEmbedding"})

    error_msg = str(exc_info.value)
    assert "Invalid Embedding name: NonExistentEmbedding" in error_msg
    assert "Available embeddings:" in error_msg
    for key in sorted(RECOGNIZED_EMBEDDINGS):
        assert key in error_msg


def test_load_embed_model_with_valid_mock() -> None:
    """MockEmbedding (always present in the registry) round-trips correctly."""
    model = load_embed_model(MockEmbedding(embed_dim=8).to_dict())
    assert isinstance(model, MockEmbedding)


def test_load_embed_model_passes_through_instance() -> None:
    """If data is already a BaseEmbedding instance, return it directly."""
    instance = MockEmbedding(embed_dim=4)
    result = load_embed_model(instance)  # type: ignore[arg-type]
    assert result is instance
