from typing import Any

from llama_index.core.callbacks.base_handler import BaseCallbackHandler


def opik_callback_handler(**eval_params: Any) -> BaseCallbackHandler:
    try:
        from opik.integrations.llama_index import LlamaIndexCallbackHandler

        return LlamaIndexCallbackHandler(**eval_params)

    except ImportError:
        raise ImportError(
            "Please install the Opik Python SDK with `pip install -U opik`"
        )
