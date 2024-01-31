from llama_index.callbacks.arize_phoenix.base import (
    arize_phoenix_callback_handlerase_handler,
)
from llama_index.core.callbacks.base_handler import BaseCallbackHandler


def test_arize_handler_callable():
    handler = arize_phoenix_callback_handlerase_handler()
    assert isinstance(handler, BaseCallbackHandler)
