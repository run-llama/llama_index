from typing import Any
from llama_index.callbacks.base_handler import BaseCallbackHandler


def arize_phoenix_callback_handler(**kwargs: Any) -> BaseCallbackHandler:
    try:
        from phoenix.trace.exporter import HttpExporter
        from phoenix.trace.llama_index import OpenInferenceTraceCallbackHandler
    except ImportError:
        raise ImportError(
            "To use the Arize Phoenix Tracer you need to "
            "have the latest `arize-phoenix` python package installed. "
            "Please install it with `pip install -q arize-phoenix`"
        )
    if "exporter" not in kwargs:
        kwargs = {"exporter": HttpExporter(), **kwargs}
    return OpenInferenceTraceCallbackHandler(**kwargs)
