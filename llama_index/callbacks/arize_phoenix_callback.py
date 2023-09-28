from typing import Any
from llama_index.callbacks.base_handler import BaseCallbackHandler


def arize_phoenix_callback_handler(**kwargs: Any) -> BaseCallbackHandler:
    try:
        from phoenix.trace.llama_index import OpenInferenceTraceCallbackHandler
        from phoenix.trace.exporter import HttpExporter
    except ImportError:
        raise ImportError(
            "Please install Arize Phoenix with `pip install -q arize-phoenix`"
        )
    if "exporter" not in kwargs:
        kwargs = {"exporter": HttpExporter(), **kwargs}
    return OpenInferenceTraceCallbackHandler(**kwargs)
