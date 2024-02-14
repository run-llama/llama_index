from llama_index.callbacks.uptrain.base import UpTrainCallbackHandler
from llama_index.core.callbacks.base_handler import BaseCallbackHandler


def test_handler_callable():
    names_of_base_classes = [b.__name__ for b in UpTrainCallbackHandler.__mro__]
    assert BaseCallbackHandler.__name__ in names_of_base_classes
