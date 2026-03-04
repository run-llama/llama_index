from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.netmind import NetmindLLM


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in NetmindLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
