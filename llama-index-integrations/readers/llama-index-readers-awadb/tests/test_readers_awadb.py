from llama_index.core.readers.base import BaseReader
from llama_index.readers.awadb import AwadbReader


def test_class():
    names_of_base_classes = [b.__name__ for b in AwadbReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
