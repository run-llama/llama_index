from typing import Any

from llama_index.legacy.callbacks.base_handler import BaseCallbackHandler


def honeyhive_callback_handler(**kwargs: Any) -> BaseCallbackHandler:
    try:
        from honeyhive.utils.llamaindex_tracer import HoneyHiveLlamaIndexTracer
    except ImportError:
        raise ImportError("Please install HoneyHive with `pip install honeyhive`")
    return HoneyHiveLlamaIndexTracer(**kwargs)
