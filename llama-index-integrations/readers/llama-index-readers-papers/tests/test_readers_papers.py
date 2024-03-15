from llama_index.core.readers.base import BaseReader
from llama_index.readers.papers import ArxivReader, PubmedReader


def test_class():
    names_of_base_classes = [b.__name__ for b in ArxivReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in PubmedReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
