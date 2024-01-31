from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.openinference.base import OpenInferenceCallbackHandler


def test_handler_callable():
    handler = OpenInferenceCallbackHandler()
    assert isinstance(handler, BaseCallbackHandler)
