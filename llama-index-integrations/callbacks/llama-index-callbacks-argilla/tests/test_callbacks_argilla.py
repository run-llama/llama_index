from llama_index.callbacks.argilla.base import argilla_callback_handler
from llama_index.core.callbacks.base_handler import BaseCallbackHandler


def test_handler_callable():
    handler = argilla_callback_handler()
    assert isinstance(handler, BaseCallbackHandler)
