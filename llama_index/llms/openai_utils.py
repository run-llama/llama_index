import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import openai
from openai import ChatCompletion, Completion
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llama_index.llms.base import ChatMessage

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

# "sk-" followed by 48 alphanumberic characters
OPENAI_API_KEY_FORMAT = re.compile("^sk-[a-zA-Z0-9]{48}$")
MISSING_API_KEY_ERROR_MESSAGE = """No API key found for OpenAI.
Please set either the OPENAI_API_KEY environment variable or \
openai.api_key prior to initialization.
API keys can be found or created at \
https://platform.openai.com/account/api-keys
"""
INVALID_API_KEY_ERROR_MESSAGE = """Invalid OpenAI API key.
API key should be of the format: "sk-" followed by \
48 alphanumeric characters.
"""

logger = logging.getLogger(__name__)

CompletionClientType = Union[Type[Completion], Type[ChatCompletion]]


def _create_retry_decorator(max_retries: int) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(is_chat_model: bool, max_retries: int, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        client = get_completion_endpoint(is_chat_model)
        return client.create(**kwargs)

    return _completion_with_retry(**kwargs)


async def acompletion_with_retry(
    is_chat_model: bool, max_retries: int, **kwargs: Any
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

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
    if "ft-" in modelname:
        modelname = modelname.split(":")[0]

    if modelname in DISCONTINUED_MODELS:
        raise ValueError(
            f"OpenAI model {modelname} has been discontinued. "
            "Please choose another model."
        )

    context_size = ALL_AVAILABLE_MODELS.get(modelname, None)

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid OpenAI model name."
            "Known models are: " + ", ".join(ALL_AVAILABLE_MODELS.keys())
        )

    return context_size


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


def to_openai_message_dict(message: ChatMessage) -> dict:
    """Convert generic message to OpenAI message dict."""
    message_dict = {
        "role": message.role,
        "content": message.content,
    }

    # NOTE: openai messages have additional arguments:
    # - function messages have `name`
    # - assistant messages have optional `function_call`
    message_dict.update(message.additional_kwargs)

    return message_dict


def to_openai_message_dicts(messages: Sequence[ChatMessage]) -> List[dict]:
    """Convert generic messages to OpenAI message dicts."""
    return [to_openai_message_dict(message) for message in messages]


def from_openai_message_dict(message_dict: dict) -> ChatMessage:
    """Convert openai message dict to generic message."""
    role = message_dict["role"]
    content = message_dict["content"]

    additional_kwargs = message_dict.copy()
    additional_kwargs.pop("role")
    additional_kwargs.pop("content")

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


def validate_openai_api_key(api_key: Optional[str] = None) -> None:
    openai_api_key = api_key or os.environ.get("OPENAI_API_KEY", "") or openai.api_key
    if not openai_api_key:
        raise ValueError(MISSING_API_KEY_ERROR_MESSAGE)
    elif not OPENAI_API_KEY_FORMAT.search(openai_api_key):
        raise ValueError(INVALID_API_KEY_ERROR_MESSAGE)
