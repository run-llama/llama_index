from llama_index.core.readers.base import BaseReader
from llama_index.readers.openalex import OpenAlexReader


def test_class():
    names_of_base_classes = [b.__name__ for b in OpenAlexReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
