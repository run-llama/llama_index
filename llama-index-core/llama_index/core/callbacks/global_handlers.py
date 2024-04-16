from typing import Any

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.simple_llm_handler import SimpleLLMHandler


def set_global_handler(eval_mode: str, **eval_params: Any) -> None:
    """Set global eval handlers."""
    import llama_index.core

    llama_index.core.global_handler = create_global_handler(eval_mode, **eval_params)


def create_global_handler(eval_mode: str, **eval_params: Any) -> BaseCallbackHandler:
    """Get global eval handler."""
    if eval_mode == "wandb":
        try:
            from llama_index.callbacks.wandb import (
                WandbCallbackHandler,
            )  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "WandbCallbackHandler is not installed. "
                "Please install it using `pip install llama-index-callbacks-wandb`"
            )

        handler: BaseCallbackHandler = WandbCallbackHandler(**eval_params)
    elif eval_mode == "openinference":
        try:
            from llama_index.callbacks.openinference import (
                OpenInferenceCallbackHandler,
            )  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "OpenInferenceCallbackHandler is not installed. "
                "Please install it using `pip install llama-index-callbacks-openinference`"
            )

        handler = OpenInferenceCallbackHandler(**eval_params)
    elif eval_mode == "arize_phoenix":
        try:
            from llama_index.callbacks.arize_phoenix import (
                arize_phoenix_callback_handler,
            )  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "ArizePhoenixCallbackHandler is not installed. "
                "Please install it using `pip install llama-index-callbacks-arize-phoenix`"
            )

        handler = arize_phoenix_callback_handler(**eval_params)
    elif eval_mode == "honeyhive":
        try:
            from llama_index.callbacks.honeyhive import (
                honeyhive_callback_handler,
            )  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "HoneyHiveCallbackHandler is not installed. "
                "Please install it using `pip install llama-index-callbacks-honeyhive`"
            )
        handler = honeyhive_callback_handler(**eval_params)
    elif eval_mode == "promptlayer":
        try:
            from llama_index.callbacks.promptlayer import (
                PromptLayerHandler,
            )  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "PromptLayerHandler is not installed. "
                "Please install it using `pip install llama-index-callbacks-promptlayer`"
            )
        handler = PromptLayerHandler(**eval_params)
    elif eval_mode == "deepeval":
        try:
            from llama_index.callbacks.deepeval import (
                deepeval_callback_handler,
            )  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "DeepEvalCallbackHandler is not installed. "
                "Please install it using `pip install llama-index-callbacks-deepeval`"
            )
        handler = deepeval_callback_handler(**eval_params)
    elif eval_mode == "simple":
        handler = SimpleLLMHandler(**eval_params)
    elif eval_mode == "argilla":
        try:
            from llama_index.callbacks.argilla import (
                argilla_callback_handler,
            )  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "ArgillaCallbackHandler is not installed. "
                "Please install it using `pip install llama-index-callbacks-argilla`"
            )
        handler = argilla_callback_handler(**eval_params)
    elif eval_mode == "langfuse":
        try:
            from llama_index.callbacks.langfuse import (
                langfuse_callback_handler,
            )  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "LangfuseCallbackHandler is not installed. "
                "Please install it using `pip install llama-index-callbacks-langfuse`"
            )
        handler = langfuse_callback_handler(**eval_params)
    else:
        raise ValueError(f"Eval mode {eval_mode} not supported.")

    return handler
