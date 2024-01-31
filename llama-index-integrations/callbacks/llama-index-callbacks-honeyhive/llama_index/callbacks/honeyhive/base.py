from typing import Any

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from honeyhive.utils.llamaindex_tracer import HoneyHiveLlamaIndexTracer


def honeyhive_callback_handler(**kwargs: Any) -> BaseCallbackHandler:
    return HoneyHiveLlamaIndexTracer(**kwargs)
