from llama_index.core.readers.base import BasePydanticReader
from llama_index.readers.slack import SlackReader


def test_class():
    names_of_base_classes = [b.__name__ for b in SlackReader.__mro__]
    assert BasePydanticReader.__name__ in names_of_base_classes
