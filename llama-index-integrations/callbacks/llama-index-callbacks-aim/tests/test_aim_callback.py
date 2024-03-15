from llama_index.callbacks.aim.base import AimCallback
from llama_index.core.callbacks.base_handler import BaseCallbackHandler


def test_class():
    names_of_base_classes = [b.__name__ for b in AimCallback.__mro__]
    assert BaseCallbackHandler.__name__ in names_of_base_classes
