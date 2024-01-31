from typing import Any

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from deepeval.tracing.integrations.llama_index import LlamaIndexCallbackHandler


def deepeval_callback_handler(**eval_params: Any) -> BaseCallbackHandler:
    return LlamaIndexCallbackHandler(**eval_params)
