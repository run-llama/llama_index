"""Tests for FireworksEmbedding integration."""

import os
from unittest.mock import patch

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.fireworks import FireworksEmbedding
from llama_index.embeddings.fireworks.base import DEFAULT_API_BASE, DEFAULT_MODEL
from llama_index.embeddings.fireworks.utils import resolve_fireworks_credentials


def test_fireworks_embedding():
    """FireworksEmbedding initializes with correct defaults."""
    emb = FireworksEmbedding(api_key="test")
    assert isinstance(emb, BaseEmbedding)
    assert FireworksEmbedding.class_name() == "FireworksEmbedding"
    assert emb.model_name == DEFAULT_MODEL
    assert emb.api_base == DEFAULT_API_BASE


def test_resolve_credentials():
    """Credentials resolve from env vars with correct defaults."""
    with patch.dict(os.environ, {}, clear=True):
        api_key, api_base, _ = resolve_fireworks_credentials()
        assert api_key == ""
        assert api_base == DEFAULT_API_BASE

    env = {"FIREWORKS_API_KEY": "env_key", "FIREWORKS_API_BASE": "https://env.api.com"}
    with patch.dict(os.environ, env, clear=True):
        api_key, api_base, _ = resolve_fireworks_credentials()
        assert api_key == "env_key"
        assert api_base == "https://env.api.com"
