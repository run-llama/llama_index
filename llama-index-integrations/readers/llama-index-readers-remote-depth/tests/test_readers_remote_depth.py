from llama_index.core.readers.base import BaseReader
from llama_index.readers.remote_depth import RemoteDepthReader


def test_class():
    names_of_base_classes = [b.__name__ for b in RemoteDepthReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
