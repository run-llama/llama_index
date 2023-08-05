"""Global eval handlers."""

from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.wandb_callback import WandbCallbackHandler
from llama_index.callbacks.open_inference_callback import OpenInferenceCallbackHandler
from typing import Any


def set_global_handler(eval_mode: str, **eval_params: Any) -> None:
    """Set global eval handlers."""

    import llama_index

    llama_index.global_handler = create_global_handler(eval_mode, **eval_params)


def create_global_handler(eval_mode: str, **eval_params: Any) -> BaseCallbackHandler:
    """Get global eval handler."""
    if eval_mode == "wandb":
        handler: BaseCallbackHandler = WandbCallbackHandler(**eval_params)
    elif eval_mode == "arize_phoenix":
        handler = OpenInferenceCallbackHandler(**eval_params)
    else:
        raise ValueError(f"Eval mode {eval_mode} not supported.")

    return handler
