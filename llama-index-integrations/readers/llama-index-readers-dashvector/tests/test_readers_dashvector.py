from llama_index.core.readers.base import BaseReader
from llama_index.readers.dashvector import DashVectorReader


def test_class():
    names_of_base_classes = [b.__name__ for b in DashVectorReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
