from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.nebius import NebiusLLM


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in NebiusLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
