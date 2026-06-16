from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.openpaths import OpenPaths
from llama_index.llms.openpaths.base import DEFAULT_API_BASE, DEFAULT_MODEL


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in OpenPaths.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_defaults() -> None:
    llm = OpenPaths(api_key="dummy_key")

    assert llm.model == DEFAULT_MODEL
    assert llm.api_base == DEFAULT_API_BASE
    assert llm.api_key == "dummy_key"
    assert llm.is_chat_model is True


def test_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENPATHS_API_KEY", "env_key")

    llm = OpenPaths(model="openpaths/auto")

    assert llm.api_key == "env_key"
