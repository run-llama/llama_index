from typing import Any

from llama_index.legacy.callbacks.base_handler import BaseCallbackHandler


def arize_phoenix_callback_handler(**kwargs: Any) -> BaseCallbackHandler:
    try:
        from phoenix.trace.llama_index import OpenInferenceTraceCallbackHandler
    except ImportError:
        raise ImportError(
            "Please install Arize Phoenix with `pip install -q arize-phoenix`"
        )
    return OpenInferenceTraceCallbackHandler(**kwargs)
