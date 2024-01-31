from llama_index.callbacks.deepeval.base import deepeval_callback_handler
from llama_index.core.callbacks.base_handler import BaseCallbackHandler


def test_handler_callable():
    handler = deepeval_callback_handler()
    assert isinstance(handler, BaseCallbackHandler)
