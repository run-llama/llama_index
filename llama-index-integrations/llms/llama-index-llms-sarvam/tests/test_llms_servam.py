from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.servam import Servam


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in Servam.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
