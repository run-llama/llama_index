"""Tests for Telnyx Embeddings integration."""

import os
from unittest.mock import patch

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.telnyx import TelnyxEmbedding


def test_telnyx_inherits_from_base_embedding():
    """Verify TelnyxEmbedding is a proper BaseEmbedding subclass."""
    names_of_base_classes = [b.__name__ for b in TelnyxEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_telnyx_class_name():
    """Verify class name."""
    assert TelnyxEmbedding.class_name() == "TelnyxEmbedding"


@patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"})
def test_telnyx_default_model():
    """Verify default model is set."""
    embed = TelnyxEmbedding()
    assert embed.model_name == "thenlper/gte-large"


@patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"})
def test_telnyx_custom_model():
    """Verify custom model can be set."""
    embed = TelnyxEmbedding(model_name="custom-model")
    assert embed.model_name == "custom-model"


@patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"})
def test_telnyx_api_base_default():
    """Verify default API base URL."""
    embed = TelnyxEmbedding()
    assert embed.api_base == "https://api.telnyx.com/v2/ai/openai"


@patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"})
def test_telnyx_api_base_custom():
    """Verify custom API base can be set."""
    embed = TelnyxEmbedding(api_base="https://custom.api.com/v1")
    assert embed.api_base == "https://custom.api.com/v1"


def test_telnyx_api_key_from_env():
    """Verify API key is read from environment."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "env-test-key"}):
        embed = TelnyxEmbedding()
        assert embed.api_key == "env-test-key"


def test_telnyx_api_key_direct():
    """Verify API key can be passed directly."""
    embed = TelnyxEmbedding(api_key="direct-test-key")
    assert embed.api_key == "direct-test-key"


def test_telnyx_api_key_missing():
    """Verify error when no API key is provided."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("TELNYX_API_KEY", None)
        with pytest.raises(ValueError, match="Telnyx API key is required"):
            TelnyxEmbedding()


@patch.dict(
    os.environ,
    {
        "TELNYX_API_KEY": "test-key",
        "TELNYX_API_BASE": "https://env-base.api.com/v1",
    },
)
def test_telnyx_api_base_from_env():
    """Verify API base can be read from environment."""
    embed = TelnyxEmbedding()
    assert embed.api_base == "https://env-base.api.com/v1"
