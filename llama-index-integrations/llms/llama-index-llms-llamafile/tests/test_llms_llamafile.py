from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.llamafile import Llamafile


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in Llamafile.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_init():
    llm = Llamafile(temperature=0.1)
    assert llm.temperature == 0.1
    assert llm.seed == 0
