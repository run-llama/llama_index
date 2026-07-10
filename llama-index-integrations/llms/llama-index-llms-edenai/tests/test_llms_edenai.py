from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.edenai import EdenAI
from llama_index.llms.edenai.base import (
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    EU_API_BASE,
)


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in EdenAI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_defaults():
    llm = EdenAI(api_key="dummy_key")
    assert llm.model == DEFAULT_MODEL
    assert llm.api_base == DEFAULT_API_BASE
    assert llm.is_chat_model is True
    assert llm.is_function_calling_model is True


def test_eu_endpoint_override():
    llm = EdenAI(api_key="dummy_key", api_base=EU_API_BASE)
    assert llm.api_base == EU_API_BASE


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("EDENAI_API_KEY", "env_key_123")
    llm = EdenAI()
    assert llm.api_key == "env_key_123"
