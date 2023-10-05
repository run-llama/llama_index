import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import openai
from openai import ChatCompletion, Completion
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llama_index.bridge.pydantic import BaseModel
from llama_index.llms.base import ChatMessage
from llama_index.llms.generic_utils import get_from_param_or_env

DEFAULT_KONKO_API_TYPE = "open_ai"
DEFAULT_KONKO_API_BASE = "https://api.konko.ai/v1"
DEFAULT_KONKO_API_VERSION = ""

LLAMA_MODELS = {
    "meta-llama/Llama-2-13b-chat-hf": 4096,
    "meta-llama/Llama-2-70b-chat-hf": 4096,
}

OPEN_AI_MODELS = {
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-3.5-turbo-0613": 4097,
    "gpt-4-0613": 8192,
    "gpt-3.5-turbo-0301": 4097,
    "gpt-3.5-turbo": 4097,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-16k-0613": 16385,
}

ALL_AVAILABLE_MODELS = {**LLAMA_MODELS, **OPEN_AI_MODELS}

DISCONTINUED_MODELS: Dict[str, int] = {}

MISSING_API_KEY_ERROR_MESSAGE = """No API key found for Konko AI.
Please set KONKO_API_KEY environment variable"""

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


def konko_modelname_to_contextsize(modelname: str) -> int:
    """Calculate the maximum number of tokens possible to generate for a model.

    Args:
        modelname: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    Example:
        .. code-block:: python

            max_tokens = konko.modelname_to_contextsize(model_name)
    """
    # handling finetuned models
    # TO BE FILLED

    if modelname in DISCONTINUED_MODELS:
        raise ValueError(
            f"Konko hosted model {modelname} has been discontinued. "
            "Please choose another model."
        )

    context_size = ALL_AVAILABLE_MODELS.get(modelname, None)

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid Konko model name."
            "Known models are: " + ", ".join(ALL_AVAILABLE_MODELS.keys())
        )

    return context_size


def is_chat_model(model: str) -> bool:
    return True


def is_function_calling_model(model: str) -> bool:
    is_chat_model_ = is_chat_model(model)
    is_old = "0314" in model or "0301" in model
    return is_chat_model_ and not is_old


def get_completion_endpoint(is_chat_model: bool) -> CompletionClientType:
    import konko

    if is_chat_model:
        return konko.ChatCompletion
    else:
        return konko.Completion


def to_openai_message_dict(message: ChatMessage) -> dict:
    """Convert generic message to OpenAI message dict."""
    message_dict = {
        "role": message.role,
        "content": message.content,
    }
    message_dict.update(message.additional_kwargs)

    return message_dict


def to_openai_message_dicts(messages: Sequence[ChatMessage]) -> List[dict]:
    """Convert generic messages to OpenAI message dicts."""
    return [to_openai_message_dict(message) for message in messages]


def from_openai_message_dict(message_dict: dict) -> ChatMessage:
    """Convert openai message dict to generic message."""
    role = message_dict["role"]
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


def resolve_konko_credentials(
    konko_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    api_type: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Tuple[str, str, str, str, str]:
    """ "Resolve KonkoAI credentials.

    The order of precedence is:
    1. param
    2. env
    3. konkoai module
    4. default
    """
    import konko

    # resolve from param or env
    konko_api_key = get_from_param_or_env(
        "konko_api_key", konko_api_key, "KONKO_API_KEY", ""
    )
    openai_api_key = get_from_param_or_env(
        "openai_api_key", openai_api_key, "OPENAI_API_KEY", ""
    )
    api_type = get_from_param_or_env("api_type", api_type, "KONKO_API_TYPE", "")
    api_base = DEFAULT_KONKO_API_BASE
    api_version = get_from_param_or_env(
        "api_version", api_version, "KONKO_API_VERSION", ""
    )

    # resolve from konko module or default
    konko_api_key = konko_api_key
    openai_api_key = openai_api_key
    api_type = api_type or DEFAULT_KONKO_API_TYPE
    api_base = api_base or konko.api_base or DEFAULT_KONKO_API_BASE
    api_version = api_version or DEFAULT_KONKO_API_VERSION

    if not konko_api_key:
        raise ValueError(MISSING_API_KEY_ERROR_MESSAGE)

    return konko_api_key, openai_api_key, api_type, api_base, api_version


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
