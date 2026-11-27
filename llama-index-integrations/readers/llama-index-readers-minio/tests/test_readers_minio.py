from llama_index.core.readers.base import BaseReader
from llama_index.readers.minio import BotoMinioReader, MinioReader


def test_class():
    names_of_base_classes = [b.__name__ for b in BotoMinioReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in MinioReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_verify_defaults_to_true():
    reader = BotoMinioReader(bucket="test-bucket")
    assert reader.verify is True


def test_verify_can_be_disabled_explicitly():
    reader = BotoMinioReader(bucket="test-bucket", verify=False)
    assert reader.verify is False
