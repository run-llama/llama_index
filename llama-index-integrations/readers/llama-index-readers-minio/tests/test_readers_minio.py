from llama_index.core.readers.base import BaseReader
from llama_index.readers.minio import BotoMinioReader, MinioReader


def test_class():
    names_of_base_classes = [b.__name__ for b in BotoMinioReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in MinioReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
