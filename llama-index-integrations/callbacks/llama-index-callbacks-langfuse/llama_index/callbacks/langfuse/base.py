from typing import Any

from llama_index.core.callbacks.base_handler import BaseCallbackHandler

from langfuse.llama_index import LlamaIndexCallbackHandler


def langfuse_callback_handler(**eval_params: Any) -> BaseCallbackHandler:
    return LlamaIndexCallbackHandler(
        **eval_params, sdk_integration="llama-index_set-global-handler"
    )
