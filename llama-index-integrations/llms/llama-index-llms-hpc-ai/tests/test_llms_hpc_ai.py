from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.hpc_ai import HpcAiLLM


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in HpcAiLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_hpc_ai_llm_class():
    model = HpcAiLLM(api_key="test")
    assert model.class_name() == "HpcAiLLM"
    assert model.model == "minimax/minimax-m2.5"


def test_hpc_ai_llm_from_env(monkeypatch):
    monkeypatch.setenv("HPC_AI_API_KEY", "env-key")
    model = HpcAiLLM()
    assert model.api_key == "env-key"


def test_hpc_ai_api_key_param_overrides_env(monkeypatch):
    monkeypatch.setenv("HPC_AI_API_KEY", "env-key")
    model = HpcAiLLM(api_key="explicit-key")
    assert model.api_key == "explicit-key"


def test_hpc_ai_api_base_from_env(monkeypatch):
    monkeypatch.setenv("HPC_AI_BASE_URL", "https://example.com/v1")
    model = HpcAiLLM(api_key="k")
    assert model.api_base == "https://example.com/v1"
