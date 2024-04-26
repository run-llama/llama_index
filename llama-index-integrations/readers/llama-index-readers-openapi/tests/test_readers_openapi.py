from llama_index.core.readers.base import BaseReader
from llama_index.readers.openapi import OpenAPIReader


def test_class():
    names_of_base_classes = [b.__name__ for b in OpenAPIReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
