from typing import Any, Dict, List, Optional, Sequence, Tuple

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

DEFAULT_ANYSCALE_API_BASE = "https://api.endpoints.anyscale.com/v1"
DEFAULT_ANYSCALE_API_VERSION = ""

LLAMA_MODELS = {
    "meta-llama/Meta-Llama-3-70B-Instruct": 8192,
    "meta-llama/Meta-Llama-3-8B-Instruct": 8192,
    "meta-llama/Llama-2-7b-chat-hf": 4096,
    "meta-llama/Llama-2-13b-chat-hf": 4096,
    "meta-llama/Llama-2-70b-chat-hf": 4096,
    "codellama/CodeLlama-34b-Instruct-hf": 16384,
    "Meta-Llama/Llama-Guard-7b": 4096,
}

MISTRAL_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.1": 16384,
    "Open-Orca/Mistral-7B-OpenOrca": 8192,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
}

ZEPHYR_MODELS = {
    "HuggingFaceH4/zephyr-7b-beta": 16384,
}

ALL_AVAILABLE_MODELS = {
    **LLAMA_MODELS,
    **MISTRAL_MODELS,
    **ZEPHYR_MODELS,
}

DISCONTINUED_MODELS: Dict[str, int] = {}


def anyscale_modelname_to_contextsize(modelname: str) -> int:
    """
    Calculate the maximum number of tokens possible to generate for a model.

    Args:
        modelname: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    Example:
        .. code-block:: python

            max_tokens = anyscale_modelname_to_contextsize(model_name)
    """
    # handling finetuned models
    # TO BE FILLED

    if modelname in DISCONTINUED_MODELS:
        raise ValueError(
            f"Anyscale hosted model {modelname} has been discontinued. "
            "Please choose another model."
        )

    context_size = ALL_AVAILABLE_MODELS.get(modelname, None)

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid Anyscale model name."
            "Known models are: " + ", ".join(ALL_AVAILABLE_MODELS.keys())
        )

    return context_size


def _message_to_anyscale_prompt(message: ChatMessage) -> Dict[str, Any]:
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


def messages_to_anyscale_prompt(messages: Sequence[ChatMessage]) -> List[Dict]:
    if len(messages) == 0:
        raise ValueError("Got empty list of messages.")

    return [_message_to_anyscale_prompt(message) for message in messages]


def resolve_anyscale_credentials(
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
    api_key = get_from_param_or_env("api_key", api_key, "ANYSCALE_API_KEY", "")
    api_base = get_from_param_or_env("api_base", api_base, "ANYSCALE_API_BASE", "")
    api_version = get_from_param_or_env(
        "api_version", api_version, "ANYSCALE_API_VERSION", ""
    )

    # resolve from openai module or default
    final_api_key = api_key or ""
    final_api_base = api_base or DEFAULT_ANYSCALE_API_BASE
    final_api_version = api_version or DEFAULT_ANYSCALE_API_VERSION

    return final_api_key, str(final_api_base), final_api_version
