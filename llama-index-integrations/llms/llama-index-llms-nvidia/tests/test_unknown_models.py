import pytest
from unittest.mock import patch, MagicMock
from llama_index.llms.nvidia import NVIDIA


@pytest.mark.parametrize("is_chat_model", [True, False])
def test_unknown_model_is_chat_model_parameter(is_chat_model) -> None:
    """Test that is_chat_model parameter is respected for unknown models."""
    mock_model = MagicMock()
    mock_model.id = "nvidia/llama-3.3-nemotron-super-49b-v2"

    with patch.object(NVIDIA, "available_models", [mock_model]):
        llm = NVIDIA(
            model="nvidia/llama-3.3-nemotron-super-49b-v2",
            is_chat_model=is_chat_model,
            api_key="fake-key",
        )
        assert llm.is_chat_model is is_chat_model


def test_unknown_model_default_is_chat_model() -> None:
    """Test that default (no parameter) defaults to False for unknown models."""
    mock_model = MagicMock()
    mock_model.id = "nvidia/llama-3.3-nemotron-super-49b-v2"

    with patch.object(NVIDIA, "available_models", [mock_model]):
        llm = NVIDIA(model="nvidia/llama-3.3-nemotron-super-49b-v2", api_key="fake-key")
        # Should default to False for unknown models
        assert llm.is_chat_model is False


def test_known_model_not_overridden() -> None:
    """Test that known models are not overridden by user-provided is_chat_model parameter."""
    mock_model = MagicMock()
    mock_model.id = "mistralai/mistral-7b-instruct-v0.2"

    with patch.object(NVIDIA, "available_models", [mock_model]):
        llm = NVIDIA(
            model="mistralai/mistral-7b-instruct-v0.2",
            is_chat_model=False,
            api_key="fake-key",
        )
        # Should not be overridden - known models keep their original setting
        assert llm.is_chat_model is True
