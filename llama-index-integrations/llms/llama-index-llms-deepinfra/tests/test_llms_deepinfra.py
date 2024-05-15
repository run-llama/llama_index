from llama_index.llms.deepinfra import DeepInfraLLM
from llama_index.core.base.llms.base import BaseLLM


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in DeepInfraLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
