from llama_index.core.readers.base import BaseReader
from llama_index.readers.iceberg import IcebergReader


def test_class():
    names_of_base_classes = [b.__name__ for b in IcebergReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
