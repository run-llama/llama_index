import warnings

warnings.warn(
    "llama-index-callbacks-langfuse is deprecated. "
    "Migrate to langfuse>=4.7 with the instrumentation module. "
    "See https://docs.llamaindex.ai/en/stable/examples/observability/Langfuse-Instrumentation/",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.callbacks.langfuse.base import langfuse_callback_handler

__all__ = ["langfuse_callback_handler"]
