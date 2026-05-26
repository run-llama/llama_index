"""Tests for Telnyx LLM integration."""

import os
from unittest.mock import patch

import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.telnyx import Telnyx


def test_telnyx_inherits_from_base_llm():
    """Verify Telnyx is a proper BaseLLM subclass."""
    names_of_base_classes = [b.__name__ for b in Telnyx.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_telnyx_class_name():
    """Verify class name."""
    assert Telnyx.class_name() == "Telnyx"


@patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"})
def test_telnyx_default_model():
    """Verify default model is set."""
    llm = Telnyx()
    assert llm.model == "meta-llama/Llama-3.3-70B-Instruct"


@patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"})
def test_telnyx_custom_model():
    """Verify custom model can be set."""
    llm = Telnyx(model="Qwen/Qwen3-235B-A22B")
    assert llm.model == "Qwen/Qwen3-235B-A22B"


@patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"})
def test_telnyx_api_base_default():
    """Verify default API base URL."""
    llm = Telnyx()
    assert llm.api_base == "https://api.telnyx.com/v2/ai/openai"


@patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"})
def test_telnyx_api_base_custom():
    """Verify custom API base can be set."""
    llm = Telnyx(api_base="https://custom.api.com/v1")
    assert llm.api_base == "https://custom.api.com/v1"


@patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"})
def test_telnyx_is_function_calling_model():
    """Verify function calling flag is set."""
    llm = Telnyx()
    assert llm.metadata.is_function_calling_model is True


def test_telnyx_api_key_from_env():
    """Verify API key is read from environment."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "env-test-key"}):
        llm = Telnyx()
        assert llm.api_key == "env-test-key"


def test_telnyx_api_key_direct():
    """Verify API key can be passed directly."""
    llm = Telnyx(api_key="direct-test-key")
    assert llm.api_key == "direct-test-key"


def test_telnyx_api_key_missing():
    """Verify error when no API key is provided."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("TELNYX_API_KEY", None)
        with pytest.raises(ValueError, match="Telnyx API key is required"):
            Telnyx()


@patch.dict(
    os.environ,
    {
        "TELNYX_API_KEY": "test-key",
        "TELNYX_API_BASE": "https://env-base.api.com/v1",
    },
)
def test_telnyx_api_base_from_env():
    """Verify API base can be read from environment."""
    llm = Telnyx()
    assert llm.api_base == "https://env-base.api.com/v1"
