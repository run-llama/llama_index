from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.watsonx import WatsonX


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in WatsonX.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
