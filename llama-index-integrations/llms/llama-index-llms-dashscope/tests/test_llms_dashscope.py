from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.dashscope import DashScope


def test_class():
    names_of_base_classes = [b.__name__ for b in DashScope.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
