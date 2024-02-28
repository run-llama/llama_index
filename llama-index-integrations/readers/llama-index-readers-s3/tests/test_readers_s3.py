from llama_index.core.readers.base import BaseReader
from llama_index.readers.s3 import S3Reader


def test_class():
    names_of_base_classes = [b.__name__ for b in S3Reader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
