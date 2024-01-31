from llama_index.callbacks.honeyhive.base import honeyhive_callback_handler
from llama_index.core.callbacks.base_handler import BaseCallbackHandler


def test_handler_callable():
    handler = honeyhive_callback_handler()
    assert isinstance(handler, BaseCallbackHandler)
