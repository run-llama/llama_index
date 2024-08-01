from llama_index.core.readers.base import BaseReader
from llama_index.readers.mondaydotcom import MondayReader


def test_class():
    names_of_base_classes = [b.__name__ for b in MondayReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
