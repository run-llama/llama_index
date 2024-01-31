from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.promptlayer.base import PromptLayerHandler


def test_handler_callable():
    handler = PromptLayerHandler()
    assert isinstance(handler, BaseCallbackHandler)
