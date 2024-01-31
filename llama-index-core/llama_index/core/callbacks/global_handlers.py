"""Global eval handlers."""

from typing import Any, Optional, Protocol

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.simple_llm_handler import SimpleLLMHandler


class HandlerCallable(Protocol):
    def __call__(self, **eval_params: Any) -> BaseCallbackHandler:
        ...


def set_global_handler(
    handler_callable: Optional[HandlerCallable], **eval_params: Any
) -> None:
    """Set global eval handlers."""
    import llama_index.core

    if handler_callable is None:
        llama_index.core.global_handler = SimpleLLMHandler(**eval_params)
    else:
        llama_index.core.global_handler = handler_callable(**eval_params)
