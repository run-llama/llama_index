from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.featherlessai import FeatherlessLLM


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in FeatherlessLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
