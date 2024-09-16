from llama_index.core.readers.base import BaseReader
from llama_index.readers.facebook_messanger import FacebookMessengerLoader


def test_class():
    names_of_base_classes = [b.__name__ for b in FacebookMessengerLoader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
