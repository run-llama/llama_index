import pytest

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.llms.sarvam import Sarvam
from llama_index.llms.sarvam.base import SARVAM_MODELS


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in Sarvam.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_defaults():
    llm = Sarvam(api_key="test-key")

    assert llm.model == "sarvam-105b"
    assert llm.api_base == "https://api.sarvam.ai/v2"
    assert llm.context_window == 128_000
    assert llm.max_tokens is None
    assert llm.metadata.context_window == 128_000
    assert llm.metadata.num_output == 128_000
    assert llm.metadata.is_chat_model is True
    assert llm.metadata.is_function_calling_model is True
    assert llm.metadata.model_name == "sarvam-105b"


@pytest.mark.parametrize("model", list(SARVAM_MODELS))
def test_model_metadata(model):
    llm = Sarvam(model=model, api_key="test-key")
    model_info = SARVAM_MODELS[model]

    assert llm.context_window == model_info.context_window
    assert llm.metadata.context_window == model_info.context_window
    assert llm.metadata.num_output == model_info.max_output_tokens
    assert llm.metadata.is_function_calling_model is model_info.supports_tools


def test_unknown_model_falls_back():
    llm = Sarvam(model="unknown-model", api_key="test-key")

    assert llm.context_window == DEFAULT_CONTEXT_WINDOW
    assert llm.metadata.context_window == DEFAULT_CONTEXT_WINDOW
    assert llm.metadata.num_output == -1
    assert llm.metadata.is_function_calling_model is False


def test_explicit_context_window_and_max_tokens_win():
    llm = Sarvam(
        model="glm5.3",
        api_key="test-key",
        context_window=4096,
        max_tokens=256,
    )

    assert llm.context_window == 4096
    assert llm.metadata.context_window == 4096
    assert llm.metadata.num_output == 256


def test_reasoning_effort_passthrough():
    llm = Sarvam(model="sarvam-105b", api_key="test-key", reasoning_effort="none")

    assert llm.reasoning_effort == "none"
    assert llm.additional_kwargs["reasoning_effort"] == "none"


def test_additional_kwargs_are_preserved():
    llm = Sarvam(
        model="glm5.3",
        api_key="test-key",
        additional_kwargs={"top_p": 0.9},
    )

    assert llm.additional_kwargs["top_p"] == 0.9


def test_env_api_key_and_base(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "env-key")
    monkeypatch.setenv("SARVAM_API_BASE", "https://example.com/v2")

    llm = Sarvam()

    assert llm.api_key == "env-key"
    assert llm.api_base == "https://example.com/v2"


def test_class_name():
    assert Sarvam.class_name() == "Sarvam_LLM"
