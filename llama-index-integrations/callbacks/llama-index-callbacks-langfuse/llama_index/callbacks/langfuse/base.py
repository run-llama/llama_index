import warnings
from typing import Any

from llama_index.core.callbacks.base_handler import BaseCallbackHandler

_DEPRECATION_MSG = (
    "llama-index-callbacks-langfuse is deprecated and will be removed in a future release. "
    "Use langfuse>=4.7 with the instrumentation module instead:\n"
    "  pip install langfuse>=4.7 openinference-instrumentation-llama-index\n"
    "  from openinference.instrumentation.llama_index import LlamaIndexInstrumentor\n"
    "  LlamaIndexInstrumentor().instrument()\n"
    "\n"
    "NOTE: The instrumentor emits OpenTelemetry spans with OpenInference attributes, which "
    "are structurally different from the legacy callback events. Dashboards, alerts, and "
    "audit queries built against the old event shape may need updating.\n"
    "\n"
    "Do NOT run both the callback and the instrumentor simultaneously — this produces "
    "duplicate traces.\n"
    "\n"
    "See https://docs.llamaindex.ai/en/stable/examples/observability/Langfuse-Instrumentation/"
)

try:
    from langfuse.llama_index import LlamaIndexCallbackHandler
except ImportError:
    LlamaIndexCallbackHandler = None


def langfuse_callback_handler(**eval_params: Any) -> BaseCallbackHandler:
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    if LlamaIndexCallbackHandler is None:
        raise ImportError(
            "langfuse package is required for the deprecated callback handler. "
            "Install it with `pip install langfuse` or migrate to the "
            "instrumentation-based integration (see README)."
        )
    return LlamaIndexCallbackHandler(
        **eval_params, sdk_integration="llama-index_set-global-handler"
    )
