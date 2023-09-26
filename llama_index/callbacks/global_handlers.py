"""Global eval handlers."""

from typing import Any

from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.open_inference_callback import OpenInferenceCallbackHandler
from llama_index.callbacks.simple_llm_handler import SimpleLLMHandler
from llama_index.callbacks.wandb_callback import WandbCallbackHandler


def set_global_handler(eval_mode: str, **eval_params: Any) -> None:
    """Set global eval handlers."""

    import llama_index

    llama_index.global_handler = create_global_handler(eval_mode, **eval_params)


def create_global_handler(eval_mode: str, **eval_params: Any) -> BaseCallbackHandler:
    """Get global eval handler."""
    if eval_mode == "wandb":
        handler: BaseCallbackHandler = WandbCallbackHandler(**eval_params)
    elif eval_mode == "openinference":
        handler = OpenInferenceCallbackHandler(**eval_params)
    elif eval_mode == "arize_phoenix":
        try:
            from phoenix.trace.llama_index import OpenInferenceTraceCallbackHandler
        except ImportError:
            raise ImportError(
                "To use the OpenInference Tracer you need to "
                "have the latest `phoenix` python package installed. "
                "Please install it with `pip install -q arize-phoenix`"
            )
        if "exporter" not in eval_params:
            from phoenix.trace.exporter import HttpExporter

            eval_params = {
                "exporter": HttpExporter(),
                **eval_params,
            }
        handler = OpenInferenceTraceCallbackHandler(**eval_params)
    elif eval_mode == "simple":
        handler = SimpleLLMHandler(**eval_params)
    else:
        raise ValueError(f"Eval mode {eval_mode} not supported.")

    return handler
