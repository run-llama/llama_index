import logging
from importlib.metadata import version
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import openai
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from packaging.version import parse
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import konko

DEFAULT_KONKO_API_TYPE = "open_ai"
DEFAULT_KONKO_API_BASE = "https://api.konko.ai/v1"
DEFAULT_KONKO_API_VERSION = ""
MISSING_API_KEY_ERROR_MESSAGE = """No Konko API key found for LLM.
E.g. to use konko Please set the KONKO_API_KEY environment variable or \
konko.api_key prior to initialization.
API keys can be found or created at \
https://www.konko.ai/
"""

logger = logging.getLogger(__name__)


def is_openai_v1() -> bool:
    try:
        _version = parse(version("openai"))
        major_version = _version.major
    except AttributeError:
        # Handle the case where version or major attribute is not present
        return False
    return bool(major_version >= 1)


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
            retry_if_exception_type(openai.APITimeoutError)
            | retry_if_exception_type(openai.APIError)
            | retry_if_exception_type(openai.APIConnectionError)
            | retry_if_exception_type(openai.RateLimitError)
            | retry_if_exception_type(openai.APIStatusError)
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


def get_completion_endpoint(is_chat_model: bool) -> Any:
    """
    Get the appropriate completion endpoint based on the model type and API version.

    Args:
    - is_chat_model (bool): A flag indicating whether the model is a chat model.

    Returns:
    - The appropriate completion endpoint based on the model type and API version.

    Raises:
    - NotImplementedError: If the combination of is_chat_model and API version is not supported.

    """
    # For OpenAI version 1
    if is_openai_v1():
        return konko.chat.completions if is_chat_model else konko.completions

    # For other versions
    if not is_openai_v1():
        return konko.ChatCompletion if is_chat_model else konko.Completion

    # Raise error if the combination of is_chat_model and API version is not covered
    raise NotImplementedError(
        "The combination of model type and API version is not supported."
    )


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


def from_openai_message_dict(message_dict: Any) -> ChatMessage:
    """Convert openai message dict to generic message."""
    if is_openai_v1():
        # Handling for OpenAI version 1
        role = message_dict.role
        content = message_dict.content
        additional_kwargs = {
            attr: getattr(message_dict, attr)
            for attr in dir(message_dict)
            if not attr.startswith("_") and attr not in ["role", "content"]
        }
    else:
        # Handling for OpenAI version 0
        role = message_dict.get("role")
        content = message_dict.get("content", None)
        additional_kwargs = {
            key: value
            for key, value in message_dict.items()
            if key not in ["role", "content"]
        }

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
    """
    "Resolve KonkoAI credentials.

    The order of precedence is:
    1. param
    2. env
    3. konkoai module
    4. default
    """
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
        if is_chat_model:
            if is_openai_v1():
                return await konko.AsyncKonko().chat.completions.create(**kwargs)
            else:
                return await konko.ChatCompletion.acreate(**kwargs)
        else:
            if is_openai_v1():
                return await konko.AsyncKonko().completions.create(**kwargs)
            else:
                return await konko.Completion.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)
