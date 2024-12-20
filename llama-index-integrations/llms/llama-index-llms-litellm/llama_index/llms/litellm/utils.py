import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import BaseModel
from openai.resources import Completions
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import litellm
from litellm.utils import Message

MISSING_API_KEY_ERROR_MESSAGE = """No API key found for LLM.
E.g. to use openai Please set the OPENAI_API_KEY environment variable or \
openai.api_key prior to initialization.
API keys can be found or created at \
https://platform.openai.com/account/api-keys
"""
INVALID_API_KEY_ERROR_MESSAGE = """Invalid LLM API key."""

logger = logging.getLogger(__name__)

CompletionClientType = Type[Completions]


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
            retry_if_exception_type(litellm.exceptions.Timeout)
            | retry_if_exception_type(litellm.exceptions.APIError)
            | retry_if_exception_type(litellm.exceptions.APIConnectionError)
            | retry_if_exception_type(litellm.exceptions.RateLimitError)
            | retry_if_exception_type(litellm.exceptions.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(is_chat_model: bool, max_retries: int, **kwargs: Any) -> Any:
    from litellm import completion

    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        return completion(**kwargs)

    return _completion_with_retry(**kwargs)


async def acompletion_with_retry(
    is_chat_model: bool, max_retries: int, **kwargs: Any
) -> Any:
    from litellm import acompletion

    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        # Use OpenAI's async api https://github.com/openai/openai-python#async-api
        return await acompletion(**kwargs)

    return await _completion_with_retry(**kwargs)


def openai_modelname_to_contextsize(modelname: str) -> int:
    import litellm

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

    try:
        context_size = int(litellm.get_max_tokens(modelname))
    except Exception:
        context_size = 2048  # by default assume models have at least 2048 tokens

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid OpenAI model name."
            "Known models are: "
            + ", ".join(litellm.model_list)
            + "\nKnown providers are: "
            + ", ".join(litellm.provider_list)
        )

    return context_size


def is_chat_model(model: str) -> bool:
    import litellm

    return model in litellm.model_list


def is_function_calling_model(model: str) -> bool:
    is_chat_model_ = is_chat_model(model)
    is_old = "0314" in model or "0301" in model
    return is_chat_model_ and not is_old


def get_completion_endpoint(is_chat_model: bool) -> CompletionClientType:
    from litellm import completion

    return completion


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
    # NOTE: Azure OpenAI returns function calling messages without a content key
    content = message_dict.get("content", None)

    additional_kwargs = message_dict.copy()
    additional_kwargs.pop("role")
    additional_kwargs.pop("content", None)

    return ChatMessage(role=role, content=content, additional_kwargs=additional_kwargs)


def from_litellm_message(message: Message) -> ChatMessage:
    """Convert litellm.utils.Message instance to generic message."""
    role = message.get("role")
    # NOTE: Azure OpenAI returns function calling messages without a content key
    content = message.get("content", None)

    return ChatMessage(role=role, content=content)


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


def validate_litellm_api_key(
    api_key: Optional[str] = None, api_type: Optional[str] = None
) -> None:
    import litellm

    api_key = litellm.validate_environment()
    if api_key is None:
        raise ValueError(MISSING_API_KEY_ERROR_MESSAGE)
