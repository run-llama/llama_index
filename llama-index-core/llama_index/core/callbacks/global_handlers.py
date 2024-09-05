from typing import Any, Optional
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.simple_llm_handler import SimpleLLMHandler


def set_global_handler(eval_mode: str, **eval_params: Any) -> None:
    """Set global eval handlers."""
    import llama_index.core

    handler = create_global_handler(eval_mode, **eval_params)
    if handler:
        llama_index.core.global_handler = handler


def create_global_handler(
    eval_mode: str, **eval_params: Any
) -> Optional[BaseCallbackHandler]:
    """Get global eval handler."""
    handler: Optional[BaseCallbackHandler] = None
    if eval_mode == "wandb":
        try:
            from llama_index.callbacks.wandb import (
                WandbCallbackHandler,
            )  # pants: no-infer-dep
        except ImportError as exc:
            raise ImportError('WandbCallbackHandler is not installed. Please install it using `pip install llama-index-callbacks-wandb`') from exc

        handler = WandbCallbackHandler(**eval_params)
    elif eval_mode == "openinference":
        try:
            from llama_index.callbacks.openinference import (
                OpenInferenceCallbackHandler,
            )  # pants: no-infer-dep
        except ImportError as exc:
            raise ImportError('OpenInferenceCallbackHandler is not installed. Please install it using `pip install llama-index-callbacks-openinference`') from exc

        handler = OpenInferenceCallbackHandler(**eval_params)
    elif eval_mode == "arize_phoenix":
        try:
            from llama_index.callbacks.arize_phoenix import (
                ArizePhoenixCallbackHandler,
            )  # pants: no-infer-dep
        except ImportError as exc:
            raise ImportError('ArizePhoenixCallbackHandler is not installed. Please install it using `pip install llama-index-callbacks-arize-phoenix`') from exc

        handler = ArizePhoenixCallbackHandler(**eval_params)
    elif eval_mode == "honeyhive":
        try:
            from llama_index.callbacks.honeyhive import (
                honeyhive_callback_handler,
            )  # pants: no-infer-dep
        except ImportError as exc:
            raise ImportError('HoneyHiveCallbackHandler is not installed. Please install it using `pip install llama-index-callbacks-honeyhive`') from exc
        handler = honeyhive_callback_handler(**eval_params)
    elif eval_mode == "promptlayer":
        try:
            from llama_index.callbacks.promptlayer import (
                PromptLayerHandler,
            )  # pants: no-infer-dep
        except ImportError as exc:
            raise ImportError('PromptLayerHandler is not installed. Please install it using `pip install llama-index-callbacks-promptlayer`') from exc
        handler = PromptLayerHandler(**eval_params)
    elif eval_mode == "deepeval":
        try:
            from llama_index.callbacks.deepeval import (
                deepeval_callback_handler,
            )  # pants: no-infer-dep
        except ImportError as exc:
            raise ImportError('DeepEvalCallbackHandler is not installed. Please install it using `pip install llama-index-callbacks-deepeval`') from exc
        handler = deepeval_callback_handler(**eval_params)
    elif eval_mode == "simple":
        handler = SimpleLLMHandler(**eval_params)
    elif eval_mode == "argilla":
        try:
            from llama_index.callbacks.argilla import (
                argilla_callback_handler,
            )  # pants: no-infer-dep
        except ImportError as exc:
            raise ImportError('ArgillaCallbackHandler is not installed. Please install it using `pip install llama-index-callbacks-argilla`') from exc
        handler = argilla_callback_handler(**eval_params)
    elif eval_mode == "langfuse":
        try:
            from llama_index.callbacks.langfuse import (
                langfuse_callback_handler,
            )  # pants: no-infer-dep
        except ImportError as exc:
            raise ImportError('LangfuseCallbackHandler is not installed. Please install it using `pip install llama-index-callbacks-langfuse`') from exc
        handler = langfuse_callback_handler(**eval_params)
    elif eval_mode == "agentops":
        try:
            from llama_index.callbacks.agentops import (
                AgentOpsHandler,
            )  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "AgentOpsHandler is not installed. "
                "Please install it using `pip install llama-index-instrumentation-agentops`"
            )
        AgentOpsHandler.init(**eval_params)
    elif eval_mode == "literalai":
        try:
            from llama_index.callbacks.literalai import (
                literalai_callback_handler,
            )  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "Literal AI Handler is not installed. "
                "Please install it using `pip install llama-index-callbacks-literalai`"
            )
        handler = literalai_callback_handler(**eval_params)
    else:
        raise ValueError(f"Eval mode {eval_mode} not supported.")

    return handler
