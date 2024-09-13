import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from deprecated import deprecated
from llama_index.core.base.llms.types import ChatMessage, LogProb, CompletionResponse
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
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

import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.completion_choice import Logprobs
from openai.types.completion import Completion

DEFAULT_OPENAI_API_TYPE = "open_ai"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_OPENAI_API_VERSION = ""

O1_MODELS: Dict[str, int] = {
    "o1-preview": 128000,
    "o1-mini": 128000,
}

GPT4_MODELS: Dict[str, int] = {
    # stable model names:
    #   resolves to gpt-4-0314 before 2023-06-27,
    #   resolves to gpt-4-0613 after
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    # turbo models (Turbo, JSON mode)
    "gpt-4-1106-preview": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-turbo-preview": 128000,
    # multimodal model
    "gpt-4-vision-preview": 128000,
    "gpt-4-1106-vision-preview": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    # 0613 models (function calling):
    #   https://openai.com/blog/function-calling-and-other-api-updates
    "gpt-4-0613": 8192,
    "gpt-4-32k-0613": 32768,
    # 0314 models
    "gpt-4-0314": 8192,
    "gpt-4-32k-0314": 32768,
}

AZURE_TURBO_MODELS: Dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-35-turbo-16k": 16384,
    "gpt-35-turbo": 4096,
    # 0125 (2024) model (JSON mode)
    "gpt-35-turbo-0125": 16384,
    # 1106 model (JSON mode)
    "gpt-35-turbo-1106": 16384,
    # 0613 models (function calling):
    "gpt-35-turbo-0613": 4096,
    "gpt-35-turbo-16k-0613": 16384,
}

TURBO_MODELS: Dict[str, int] = {
    # stable model names:
    #   resolves to gpt-3.5-turbo-0125 as of 2024-04-29.
    "gpt-3.5-turbo": 16384,
    # resolves to gpt-3.5-turbo-16k-0613 until 2023-12-11
    # resolves to gpt-3.5-turbo-1106 after
    "gpt-3.5-turbo-16k": 16384,
    # 0125 (2024) model (JSON mode)
    "gpt-3.5-turbo-0125": 16384,
    # 1106 model (JSON mode)
    "gpt-3.5-turbo-1106": 16384,
    # 0613 models (function calling):
    #   https://openai.com/blog/function-calling-and-other-api-updates
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k-0613": 16384,
    # 0301 models
    "gpt-3.5-turbo-0301": 4096,
}

GPT3_5_MODELS: Dict[str, int] = {
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    # instruct models
    "gpt-3.5-turbo-instruct": 4096,
}

GPT3_MODELS: Dict[str, int] = {
    "text-ada-001": 2049,
    "text-babbage-001": 2040,
    "text-curie-001": 2049,
    "ada": 2049,
    "babbage": 2049,
    "curie": 2049,
    "davinci": 2049,
}

ALL_AVAILABLE_MODELS = {
    **O1_MODELS,
    **GPT4_MODELS,
    **TURBO_MODELS,
    **GPT3_5_MODELS,
    **GPT3_MODELS,
    **AZURE_TURBO_MODELS,
}

CHAT_MODELS = {
    **O1_MODELS,
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

OpenAIToolCall = Union[ChatCompletionMessageToolCall, ChoiceDeltaToolCall]


def create_retry_decorator(
    max_retries: int,
    random_exponential: bool = False,
    stop_after_delay_seconds: Optional[float] = None,
    min_seconds: float = 4,
    max_seconds: float = 60,
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
                    openai.APIConnectionError,
                    openai.APITimeoutError,
                    openai.RateLimitError,
                    openai.InternalServerError,
                )
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


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

    # TODO: This is temporary for openai's beta
    is_o1_beta = "o1" in model

    return is_chat_model_ and not is_old and not is_o1_beta


def to_openai_message_dict(
    message: ChatMessage, drop_none: bool = False, model: Optional[str] = None
) -> ChatCompletionMessageParam:
    """Convert generic message to OpenAI message dict."""
    message_dict = {
        "role": message.role.value,
        "content": message.content,
    }

    # TODO: O1 models do not support system prompts
    if model is not None and model in O1_MODELS:
        if message_dict["role"] == "system":
            message_dict["role"] = "user"

    # NOTE: openai messages have additional arguments:
    # - function messages have `name`
    # - assistant messages have optional `function_call`
    message_dict.update(message.additional_kwargs)

    null_keys = [key for key, value in message_dict.items() if value is None]
    # if drop_none is True, remove keys with None values
    if drop_none:
        for key in null_keys:
            message_dict.pop(key)

    return message_dict  # type: ignore


def to_openai_message_dicts(
    messages: Sequence[ChatMessage],
    drop_none: bool = False,
    model: Optional[str] = None,
) -> List[ChatCompletionMessageParam]:
    """Convert generic messages to OpenAI message dicts."""
    return [
        to_openai_message_dict(message, drop_none=drop_none, model=model)
        for message in messages
    ]


def from_openai_message(openai_message: ChatCompletionMessage) -> ChatMessage:
    """Convert openai message dict to generic message."""
    role = openai_message.role
    # NOTE: Azure OpenAI returns function calling messages without a content key
    content = openai_message.content

    # function_call = None  # deprecated in OpenAI v 1.1.0

    additional_kwargs: Dict[str, Any] = {}
    if openai_message.tool_calls is not None:
        tool_calls: List[ChatCompletionMessageToolCall] = openai_message.tool_calls
        additional_kwargs.update(tool_calls=tool_calls)

    return ChatMessage(role=role, content=content, additional_kwargs=additional_kwargs)


def from_openai_token_logprob(
    openai_token_logprob: ChatCompletionTokenLogprob,
) -> List[LogProb]:
    """Convert a single openai token logprob to generic list of logprobs."""
    result = []
    if openai_token_logprob.top_logprobs:
        try:
            result = [
                LogProb(token=el.token, logprob=el.logprob, bytes=el.bytes or [])
                for el in openai_token_logprob.top_logprobs
            ]
        except Exception as e:
            print(openai_token_logprob)
            raise
    return result


def from_openai_token_logprobs(
    openai_token_logprobs: Sequence[ChatCompletionTokenLogprob],
) -> List[List[LogProb]]:
    """Convert openai token logprobs to generic list of LogProb."""
    result = []
    for token_logprob in openai_token_logprobs:
        if logprobs := from_openai_token_logprob(token_logprob):
            result.append(logprobs)
    return result


def from_openai_completion_logprob(
    openai_completion_logprob: Dict[str, float]
) -> List[LogProb]:
    """Convert openai completion logprobs to generic list of LogProb."""
    return [
        LogProb(token=t, logprob=v, bytes=[])
        for t, v in openai_completion_logprob.items()
    ]


def from_openai_completion_logprobs(
    openai_completion_logprobs: Logprobs,
) -> List[List[LogProb]]:
    """Convert openai completion logprobs to generic list of LogProb."""
    result = []
    if openai_completion_logprobs.top_logprobs:
        result = [
            from_openai_completion_logprob(completion_logprob)
            for completion_logprob in openai_completion_logprobs.top_logprobs
        ]
    return result


def from_openai_completion(openai_completion: Completion) -> CompletionResponse:
    """Convert openai completion to CompletionResponse."""
    text = openai_completion.choices[0].text


def from_openai_messages(
    openai_messages: Sequence[ChatCompletionMessage],
) -> List[ChatMessage]:
    """Convert openai message dicts to generic messages."""
    return [from_openai_message(message) for message in openai_messages]


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


@deprecated("Deprecated in favor of `to_openai_tool`, which should be used instead.")
def to_openai_function(pydantic_class: Type[BaseModel]) -> Dict[str, Any]:
    """Deprecated in favor of `to_openai_tool`.

    Convert pydantic class to OpenAI function.
    """
    return to_openai_tool(pydantic_class, description=None)


def to_openai_tool(
    pydantic_class: Type[BaseModel], description: Optional[str] = None
) -> Dict[str, Any]:
    """Convert pydantic class to OpenAI tool."""
    schema = pydantic_class.schema()
    schema_description = schema.get("description", None) or description
    function = {
        "name": schema["title"],
        "description": schema_description,
        "parameters": pydantic_class.schema(),
    }
    return {"type": "function", "function": function}


def resolve_openai_credentials(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Tuple[Optional[str], str, str]:
    """ "Resolve OpenAI credentials.

    The order of precedence is:
    1. param
    2. env
    3. openai module
    4. default
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "OPENAI_API_KEY", "")
    api_base = get_from_param_or_env("api_base", api_base, "OPENAI_API_BASE", "")
    api_version = get_from_param_or_env(
        "api_version", api_version, "OPENAI_API_VERSION", ""
    )

    # resolve from openai module or default
    final_api_key = api_key or openai.api_key or ""
    final_api_base = api_base or openai.base_url or DEFAULT_OPENAI_API_BASE
    final_api_version = api_version or openai.api_version or DEFAULT_OPENAI_API_VERSION

    return final_api_key, str(final_api_base), final_api_version


def validate_openai_api_key(api_key: Optional[str] = None) -> None:
    openai_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    if not openai_api_key:
        raise ValueError(MISSING_API_KEY_ERROR_MESSAGE)
