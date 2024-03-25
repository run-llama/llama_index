from llama_index.core.readers.base import BaseReader
from llama_index.readers.google import GoogleDriveReader


def test_class():
    names_of_base_classes = [b.__name__ for b in GoogleDriveReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
