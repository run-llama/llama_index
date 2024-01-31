from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.wandb.base import WandbCallbackHandler


def test_handler_callable():
    handler = WandbCallbackHandler()
    assert isinstance(handler, BaseCallbackHandler)
