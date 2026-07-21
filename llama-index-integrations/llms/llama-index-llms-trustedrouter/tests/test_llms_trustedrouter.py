from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.trustedrouter import TrustedRouter


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in TrustedRouter.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_defaults():
    llm = TrustedRouter(api_key="test-key")
    assert llm.model == "trustedrouter/zdr"
    assert llm.api_base == "https://api.trustedrouter.com/v1"
    assert llm.is_chat_model is True


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("TRUSTEDROUTER_API_KEY", "env-key")
    llm = TrustedRouter()
    assert llm.api_key == "env-key"
