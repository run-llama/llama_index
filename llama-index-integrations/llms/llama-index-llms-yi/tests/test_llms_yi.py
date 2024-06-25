from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.yi import Yi


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in Yi.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
