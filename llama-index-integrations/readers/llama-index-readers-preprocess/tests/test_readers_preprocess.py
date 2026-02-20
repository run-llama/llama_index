import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.preprocess import PreprocessReader


def test_class():
    names_of_base_classes = [b.__name__ for b in PreprocessReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_init_raises():
    with pytest.raises(RuntimeError, match="discontinued"):
        PreprocessReader(api_key="test")
