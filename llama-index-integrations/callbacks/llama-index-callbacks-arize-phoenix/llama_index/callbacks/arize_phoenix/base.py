from typing import Any

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from phoenix.trace.exporter import HttpExporter
from phoenix.trace.llama_index import OpenInferenceTraceCallbackHandler


def arize_phoenix_callback_handler(**kwargs: Any) -> BaseCallbackHandler:
    if "exporter" not in kwargs:
        kwargs = {"exporter": HttpExporter(), **kwargs}
    return OpenInferenceTraceCallbackHandler(**kwargs)
