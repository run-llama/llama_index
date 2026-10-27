from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.everlyai import EverlyAI


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in EverlyAI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_api_key_from_env(monkeypatch):
    monkeypatch.delenv("EverlyAI_API_KEY", raising=False)
    monkeypatch.setenv("EVERLYAI_API_KEY", "upper-key")
    assert EverlyAI().api_key == "upper-key"


def test_api_key_from_legacy_env(monkeypatch):
    monkeypatch.delenv("EVERLYAI_API_KEY", raising=False)
    monkeypatch.setenv("EverlyAI_API_KEY", "legacy-key")
    assert EverlyAI().api_key == "legacy-key"


def test_api_key_param_wins(monkeypatch):
    monkeypatch.setenv("EVERLYAI_API_KEY", "upper-key")
    assert EverlyAI(api_key="param-key").api_key == "param-key"
