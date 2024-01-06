from typing import Any

from llama_index.callbacks.base_handler import BaseCallbackHandler


def deepeval_callback_handler(**kwargs: Any) -> BaseCallbackHandler:
    try:
        from deepeval.tracing.integrations.llama_index import LlamaIndexCallbackHandler
    except ImportError:
        raise ImportError("Please install DeepEval with `pip install -U deepeval`")
    return LlamaIndexCallbackHandler(**kwargs)
