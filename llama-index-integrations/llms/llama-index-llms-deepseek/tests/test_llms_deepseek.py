from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.deepseek import DeepSeek


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in DeepSeek.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
