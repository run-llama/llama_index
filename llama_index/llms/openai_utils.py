import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import openai
from openai import ChatCompletion, Completion
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random_exponential,
)
from tenacity.stop import stop_base

from llama_index.bridge.pydantic import BaseModel
from llama_index.llms.base import ChatMessage
from llama_index.llms.generic_utils import get_from_param_or_env

DEFAULT_OPENAI_API_TYPE = "open_ai"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_OPENAI_API_VERSION = ""


GPT4_MODELS = {
    # stable model names:
    #   resolves to gpt-4-0314 before 2023-06-27,
    #   resolves to gpt-4-0613 after
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    # 0613 models (function calling):
    #   https://openai.com/blog/function-calling-and-other-api-updates
    "gpt-4-0613": 8192,
    "gpt-4-32k-0613": 32768,
    # 0314 models
    "gpt-4-0314": 8192,
    "gpt-4-32k-0314": 32768,
}

AZURE_TURBO_MODELS = {
    "gpt-35-turbo-16k": 16384,
    "gpt-35-turbo": 4096,
}

TURBO_MODELS = {
    # stable model names:
    #   resolves to gpt-3.5-turbo-0301 before 2023-06-27,
    #   resolves to gpt-3.5-turbo-0613 after
    "gpt-3.5-turbo": 4096,
    # resolves to gpt-3.5-turbo-16k-0613
    "gpt-3.5-turbo-16k": 16384,
    # 0613 models (function calling):
    #   https://openai.com/blog/function-calling-and-other-api-updates
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k-0613": 16384,
    # 0301 models
    "gpt-3.5-turbo-0301": 4096,
}

GPT3_5_MODELS = {
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    # instruct models
    "gpt-3.5-turbo-instruct": 4096,
}

GPT3_MODELS = {
    "text-ada-001": 2049,
    "text-babbage-001": 2040,
    "text-curie-001": 2049,
    "ada": 2049,
    "babbage": 2049,
    "curie": 2049,
    "davinci": 2049,
}

ALL_AVAILABLE_MODELS = {
    **GPT4_MODELS,
    **TURBO_MODELS,
    **GPT3_5_MODELS,
    **GPT3_MODELS,
    **AZURE_TURBO_MODELS,
}

CHAT_MODELS = {
    **GPT4_MODELS,
    **TURBO_MODELS,
    **AZURE_TURBO_MODELS,
}


DISCONTINUED_MODELS = {
    "code-davinci-002": 8001,
    "code-davinci-001": 8001,
    "code-cushman-002": 2048,
    "code-cushman-001": 2048,
}

MISSING_API_KEY_ERROR_MESSAGE = """No API key found for OpenAI.
Please set either the OPENAI_API_KEY environment variable or \
openai.api_key prior to initialization.
API keys can be found or created at \
https://platform.openai.com/account/api-keys
"""

logger = logging.getLogger(__name__)

CompletionClientType = Union[Type[Completion], Type[ChatCompletion]]


def create_retry_decorator(
    max_retries: int,
    random_exponential: bool = False,
    stop_after_delay_seconds: Optional[float] = None,
    min_seconds: float = 4,
    max_seconds: float = 10,
) -> Callable[[Any], Any]:
    wait_strategy = (
        wait_random_exponential(min=min_seconds, max=max_seconds)
        if random_exponential
        else wait_exponential(multiplier=1, min=min_seconds, max=max_seconds)
    )

    stop_strategy: stop_base = stop_after_attempt(max_retries)
    if stop_after_delay_seconds is not None:
        stop_strategy = stop_strategy | stop_after_delay(stop_after_delay_seconds)

    return retry(
        reraise=True,
        stop=stop_strategy,
        wait=wait_strategy,
        retry=(
            retry_if_exception_type(
                (
                    openai.error.Timeout,
                    openai.error.APIError,
                    openai.error.APIConnectionError,
                    openai.error.RateLimitError,
                    openai.error.ServiceUnavailableError,
                )
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(
    is_chat_model: bool,
    max_retries: int,
    min_seconds: float = 4,
    max_seconds: float = 10,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = create_retry_decorator(
        max_retries=max_retries, min_seconds=min_seconds, max_seconds=max_seconds
    )

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        client = get_completion_endpoint(is_chat_model)
        return client.create(**kwargs)

    return _completion_with_retry(**kwargs)


async def acompletion_with_retry(
    is_chat_model: bool,
    max_retries: int,
    min_seconds: float = 4,
    max_seconds: float = 10,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = create_retry_decorator(
        max_retries=max_retries, min_seconds=min_seconds, max_seconds=max_seconds
    )

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        # Use OpenAI's async api https://github.com/openai/openai-python#async-api
        client = get_completion_endpoint(is_chat_model)
        return await client.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)


def openai_modelname_to_contextsize(modelname: str) -> int:
    """Calculate the maximum number of tokens possible to generate for a model.

    Args:
        modelname: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    Example:
        .. code-block:: python

            max_tokens = openai.modelname_to_contextsize("text-davinci-003")

    Modified from:
        https://github.com/hwchase17/langchain/blob/master/langchain/llms/openai.py
    """
    # handling finetuned models
    if modelname.startswith("ft:"):
        modelname = modelname.split(":")[1]
    elif ":ft-" in modelname:  # legacy fine-tuning
        modelname = modelname.split(":")[0]

    if modelname in DISCONTINUED_MODELS:
        raise ValueError(
            f"OpenAI model {modelname} has been discontinued. "
            "Please choose another model."
        )
    if modelname not in ALL_AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model {modelname!r}. Please provide a valid OpenAI model name in:"
            f" {', '.join(ALL_AVAILABLE_MODELS.keys())}"
        )
    return ALL_AVAILABLE_MODELS[modelname]


def is_chat_model(model: str) -> bool:
    return model in CHAT_MODELS


def is_function_calling_model(model: str) -> bool:
    is_chat_model_ = is_chat_model(model)
    is_old = "0314" in model or "0301" in model
    return is_chat_model_ and not is_old


def get_completion_endpoint(is_chat_model: bool) -> CompletionClientType:
    if is_chat_model:
        return openai.ChatCompletion
    else:
        return openai.Completion


def to_openai_message_dict(message: ChatMessage, drop_none: bool = False) -> dict:
    """Convert generic message to OpenAI message dict."""
    message_dict = {
        "role": message.role,
        "content": message.content,
    }

    # NOTE: openai messages have additional arguments:
    # - function messages have `name`
    # - assistant messages have optional `function_call`
    message_dict.update(message.additional_kwargs)

    null_keys = [key for key, value in message_dict.items() if value is None]
    # if drop_none is True, remove keys with None values
    if drop_none:
        for key in null_keys:
            message_dict.pop(key)

    return message_dict


def to_openai_message_dicts(
    messages: Sequence[ChatMessage], drop_none: bool = False
) -> List[dict]:
    """Convert generic messages to OpenAI message dicts."""
    return [
        to_openai_message_dict(message, drop_none=drop_none) for message in messages
    ]


def from_openai_message_dict(message_dict: dict) -> ChatMessage:
    """Convert openai message dict to generic message."""
    role = message_dict["role"]
    # NOTE: Azure OpenAI returns function calling messages without a content key
    content = message_dict.get("content", None)

    additional_kwargs = message_dict.copy()
    additional_kwargs.pop("role")
    additional_kwargs.pop("content", None)

    return ChatMessage(role=role, content=content, additional_kwargs=additional_kwargs)


def from_openai_message_dicts(message_dicts: Sequence[dict]) -> List[ChatMessage]:
    """Convert openai message dicts to generic messages."""
    return [from_openai_message_dict(message_dict) for message_dict in message_dicts]


def to_openai_function(pydantic_class: Type[BaseModel]) -> Dict[str, Any]:
    """Convert pydantic class to OpenAI function."""
    schema = pydantic_class.schema()
    return {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": pydantic_class.schema(),
    }


def resolve_openai_credentials(
    api_key: Optional[str] = None,
    api_type: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Tuple[str, str, str, str]:
    """ "Resolve OpenAI credentials.

    The order of precedence is:
    1. param
    2. env
    3. openai module
    4. default
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "OPENAI_API_KEY", "")
    api_type = get_from_param_or_env("api_type", api_type, "OPENAI_API_TYPE", "")
    api_base = get_from_param_or_env("api_base", api_base, "OPENAI_API_BASE", "")
    api_version = get_from_param_or_env(
        "api_version", api_version, "OPENAI_API_VERSION", ""
    )

    # resolve from openai module or default
    api_key = api_key or openai.api_key
    api_type = api_type or openai.api_type or DEFAULT_OPENAI_API_TYPE
    api_base = api_base or openai.api_base or DEFAULT_OPENAI_API_BASE
    api_version = api_version or openai.api_version or DEFAULT_OPENAI_API_VERSION

    if not api_key:
        raise ValueError(MISSING_API_KEY_ERROR_MESSAGE)

    return api_key, api_type, api_base, api_version


def validate_openai_api_key(api_key: Optional[str] = None) -> None:
    openai_api_key = api_key or os.environ.get("OPENAI_API_KEY", "") or openai.api_key

    if not openai_api_key:
        raise ValueError(MISSING_API_KEY_ERROR_MESSAGE)


def resolve_from_aliases(*args: Optional[str]) -> Optional[str]:
    for arg in args:
        if arg is not None:
            return arg
    return None
