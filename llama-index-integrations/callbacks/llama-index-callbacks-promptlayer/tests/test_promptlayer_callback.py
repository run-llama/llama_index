from llama_index.callbacks.promptlayer.base import PromptLayerHandler
from llama_index.core.callbacks.base_handler import BaseCallbackHandler


def test_handler_callable():
    names_of_base_classes = [b.__name__ for b in PromptLayerHandler.__mro__]
    assert BaseCallbackHandler.__name__ in names_of_base_classes
