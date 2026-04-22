from typing import Any, Dict, List, Optional, Sequence, Tuple

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

DEFAULT_FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1"
DEFAULT_FIREWORKS_API_VERSION = ""

SERVERLESS_MODELS = {
    "accounts/fireworks/models/deepseek-v3p1-terminus": 163840,
    "accounts/fireworks/models/kimi-k2-instruct-0905": 262144,
    "accounts/fireworks/models/glm-4p6": 202752,
    "accounts/fireworks/models/deepseek-v3p1": 163840,
    "accounts/fireworks/models/gpt-oss-120b": 131072,
    "accounts/fireworks/models/gpt-oss-20b": 131072,
    "accounts/fireworks/models/kimi-k2-instruct": 131072,
    "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507": 262144,
    "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct": 262144,
    "accounts/fireworks/models/qwen3-235b-a22b-thinking-2507": 262144,
    "accounts/fireworks/models/qwen3-235b-a22b": 131072,
    "accounts/fireworks/models/deepseek-r1-0528": 163840,
    "accounts/fireworks/models/glm-4p5": 131072,
    "accounts/fireworks/models/llama4-maverick-instruct-basic": 1048576,
    "accounts/fireworks/models/qwen2p5-vl-32b-instruct": 128000,
    "accounts/fireworks/models/llama-v3p1-8b-instruct": 131072,
    "accounts/fireworks/models/deepseek-v3-0324": 163840,
    "accounts/fireworks/models/llama-v3p3-70b-instruct": 131072,
    "accounts/fireworks/models/llama-v3p1-70b-instruct": 131072,
    "accounts/fireworks/models/llama4-scout-instruct-basic": 1048576,
    "accounts/fireworks/models/qwen3-8b": 40960,
    "accounts/fireworks/models/qwen3-embedding-8b": 40960,
    "accounts/fireworks/models/qwen3-vl-235b-a22b-instruct": 32768,
    "accounts/fireworks/models/qwen3-vl-235b-a22b-thinking": 32768,
    "accounts/fireworks/models/qwen3-reranker-8b": 40960,
    "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct": 32768,
    "accounts/fireworks/models/qwen3-vl-30b-a3b-thinking": 32768,
    "accounts/fireworks/models/minimax-m2": 131072,
}


FUNCTION_CALLING_MODELS = {
    "accounts/fireworks/models/deepseek-v3p1-terminus": 163840,
    "accounts/fireworks/models/kimi-k2-instruct-0905": 262144,
    "accounts/fireworks/models/glm-4p6": 202752,
    "accounts/fireworks/models/deepseek-v3p1": 163840,
    "accounts/fireworks/models/gpt-oss-120b": 131072,
    "accounts/fireworks/models/kimi-k2-instruct": 131072,
    "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507": 262144,
    "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct": 262144,
    "accounts/fireworks/models/qwen3-235b-a22b": 131072,
    "accounts/fireworks/models/deepseek-r1-0528": 163840,
    "accounts/fireworks/models/glm-4p5": 131072,
    "accounts/fireworks/models/llama4-maverick-instruct-basic": 1048576,
    "accounts/fireworks/models/deepseek-v3-0324": 163840,
    "accounts/fireworks/models/llama-v3p1-70b-instruct": 131072,
    "accounts/fireworks/models/llama4-scout-instruct-basic": 1048576,
    "accounts/fireworks/models/qwen3-8b": 40960,
    "accounts/fireworks/models/qwen3-vl-235b-a22b-instruct": 32768,
    "accounts/fireworks/models/qwen3-vl-235b-a22b-thinking": 32768,
    "accounts/fireworks/models/minimax-m2": 131072,
}


ALL_AVAILABLE_MODELS = {
    **SERVERLESS_MODELS,
    **FUNCTION_CALLING_MODELS,
}

DISCONTINUED_MODELS: Dict[str, int] = {}


def fireworks_modelname_to_contextsize(modelname: str) -> int:
    """
    Calculate the maximum number of tokens possible to generate for a model.

    Args:
        modelname: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    Example:
        .. code-block:: python

            max_tokens = fireworks_modelname_to_contextsize(model_name)

    """
    context_size = ALL_AVAILABLE_MODELS.get(modelname)

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid Fireworks model name."
            "Known models are: " + ", ".join(ALL_AVAILABLE_MODELS.keys())
        )

    return context_size


def is_function_calling_model(model: str) -> bool:
    """Check if a model supports function calling."""
    return model in FUNCTION_CALLING_MODELS


def _message_to_fireworks_prompt(message: ChatMessage) -> Dict[str, Any]:
    if message.role == MessageRole.USER:
        prompt = {"role": "user", "content": message.content}
    elif message.role == MessageRole.ASSISTANT:
        prompt = {"role": "assistant", "content": message.content}
    elif message.role == MessageRole.SYSTEM:
        prompt = {"role": "system", "content": message.content}
    elif message.role == MessageRole.FUNCTION:
        raise ValueError(f"Message role {MessageRole.FUNCTION} is not supported.")
    else:
        raise ValueError(f"Unknown message role: {message.role}")

    return prompt


def messages_to_fireworks_prompt(messages: Sequence[ChatMessage]) -> List[Dict]:
    if len(messages) == 0:
        raise ValueError("Got empty list of messages.")

    return [_message_to_fireworks_prompt(message) for message in messages]


def resolve_fireworks_credentials(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Tuple[Optional[str], str, str]:
    """
    "Resolve OpenAI credentials.

    The order of precedence is:
    1. param
    2. env
    3. openai module
    4. default
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "FIREWORKS_API_KEY", "")
    api_base = get_from_param_or_env("api_base", api_base, "FIREWORKS_API_BASE", "")
    api_version = get_from_param_or_env(
        "api_version", api_version, "FIREWORKS_API_VERSION", ""
    )

    # resolve from openai module or default
    final_api_key = api_key or ""
    final_api_base = api_base or DEFAULT_FIREWORKS_API_BASE
    final_api_version = api_version or DEFAULT_FIREWORKS_API_VERSION

    return final_api_key, str(final_api_base), final_api_version
