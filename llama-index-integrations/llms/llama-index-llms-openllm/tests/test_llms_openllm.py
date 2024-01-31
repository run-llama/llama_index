from llama_index.core.llms.base import BaseLLM
from llama_index.llms.openllm import OpenLLM, OpenLLMAPI


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in OpenLLMAPI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in OpenLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
