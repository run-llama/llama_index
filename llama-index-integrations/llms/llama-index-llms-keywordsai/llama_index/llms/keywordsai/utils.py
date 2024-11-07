import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from functools import lru_cache

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

DEFAULT_KEYWORDSAI_API_TYPE = "keywords_ai"
DEFAULT_KEYWORDSAI_API_BASE = "https://api.keywordsai.co/api/"
DEFAULT_KEYWORDSAI_API_VERSION = ""

MISSING_API_KEY_ERROR_MESSAGE = """No API key found for KeywordsAI.
Please set either the KEYWORDSAI_API_KEY environment variable or \
openai.api_key prior to initialization.
API keys can be found or created at \
https://platform.openai.com/account/api-keys
"""

logger = logging.getLogger(__name__)

KeywordsAIToolCall = Union[ChatCompletionMessageToolCall, ChoiceDeltaToolCall]

O1_MODELS: Dict[str, int] = {
    "o1-preview": 128000,
    "o1-preview-2024-09-12": 128000,
    "o1-mini": 128000,
    "o1-mini-2024-09-12": 128000,
}


@lru_cache
def get_keywords_models():
    """Get available models from KeywordsAI API.

    Returns:
        List of model configurations with pricing and provider info.

    Raises:
        Exception if API call fails after retries.
    """
    import requests
    from typing import Dict, List

    def _get_models() -> List[Dict]:
        response = requests.get("https://api.keywordsai.co/api/models/public")
        if not response.ok:
            raise Exception(f"Failed to fetch models: {response.status_code}")

        data = response.json()
        models = [m for m in data["models"] if m.get("input_cost")]

        # Process model data
        for model in models:
            # Convert costs from per token to per million tokens
            model["input_cost"] = model["input_cost"] * 1e6
            model["output_cost"] = model["output_cost"] * 1e6

            # Normalize Google provider names
            if model["provider"]["provider_id"] in [
                "google_palm",
                "google_vertex_ai",
                "google_gemini_ai",
            ]:
                model["provider"]["provider_id"] = "google"
                model["provider"]["provider_name"] = "Google"

        return {model["model_name"]: model for model in models}

    # Use retry decorator for resilience
    retry_decorator = create_retry_decorator(
        max_retries=3, random_exponential=True, min_seconds=1, max_seconds=10
    )

    try:
        return retry_decorator(_get_models)()
    except Exception as e:
        logger.error(f"Failed to fetch models after retries: {e!s}")
        return []


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


def keywordsai_modelname_to_contextsize(modelname: str) -> int:
    """Calculate the maximum number of tokens possible to generate for a model."""
    models = get_keywords_models()
    return models[modelname]["max_context_window"]


def is_chat_model(model: str) -> bool:
    # TODO: check if they are chat models
    return True


def is_function_calling_model(model: str) -> bool:
    models = get_keywords_models()
    return models[model]["function_call"] == 1


def to_openai_message_dict(
    message: ChatMessage, drop_none: bool = False, model: Optional[str] = None
) -> ChatCompletionMessageParam:
    """Convert generic message to KeywordsAI message dict."""
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
    """Convert generic messages to KeywordsAI message dicts."""
    return [
        to_openai_message_dict(message, drop_none=drop_none, model=model)
        for message in messages
    ]


def from_openai_message(openai_message: ChatCompletionMessage) -> ChatMessage:
    """Convert openai message dict to generic message."""
    role = openai_message.role
    # NOTE: Azure KeywordsAI returns function calling messages without a content key
    content = openai_message.content

    # function_call = None  # deprecated in KeywordsAI v 1.1.0

    additional_kwargs: Dict[str, Any] = {}
    if openai_message.tool_calls:
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
    # NOTE: Azure KeywordsAI returns function calling messages without a content key
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

    Convert pydantic class to KeywordsAI function.
    """
    return to_openai_tool(pydantic_class, description=None)


def to_openai_tool(
    pydantic_class: Type[BaseModel], description: Optional[str] = None
) -> Dict[str, Any]:
    """Convert pydantic class to KeywordsAI tool."""
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
    """ "Resolve KeywordsAI credentials.

    The order of precedence is:
    1. param
    2. env
    3. openai module
    4. default
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "KEYWORDSAI_API_KEY", "")
    api_base = get_from_param_or_env("api_base", api_base, "KEYWORDSAI_API_BASE", "")
    api_version = get_from_param_or_env(
        "api_version", api_version, "KEYWORDSAI_API_VERSION", ""
    )

    # resolve from openai module or default
    final_api_key = api_key or openai.api_key or ""
    final_api_base = api_base or openai.base_url or DEFAULT_KEYWORDSAI_API_BASE
    final_api_version = (
        api_version or openai.api_version or DEFAULT_KEYWORDSAI_API_VERSION
    )

    return final_api_key, str(final_api_base), final_api_version


def validate_openai_api_key(api_key: Optional[str] = None) -> None:
    openai_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    if not openai_api_key:
        raise ValueError(MISSING_API_KEY_ERROR_MESSAGE)


def resolve_tool_choice(tool_choice: Union[str, dict] = "auto") -> Union[str, dict]:
    """Resolve tool choice.

    If tool_choice is a function name string, return the appropriate dict.
    """
    if isinstance(tool_choice, str) and tool_choice not in ["none", "auto", "required"]:
        return {"type": "function", "function": {"name": tool_choice}}

    return tool_choice


def update_tool_calls(
    tool_calls: List[ChoiceDeltaToolCall],
    tool_calls_delta: Optional[List[ChoiceDeltaToolCall]],
) -> List[ChoiceDeltaToolCall]:
    """
    Use the tool_calls_delta objects received from openai stream chunks
    to update the running tool_calls object.

    Args:
        tool_calls (List[ChoiceDeltaToolCall]): the list of tool calls
        tool_calls_delta (ChoiceDeltaToolCall): the delta to update tool_calls

    Returns:
        List[ChoiceDeltaToolCall]: the updated tool calls
    """
    # openai provides chunks consisting of tool_call deltas one tool at a time
    if tool_calls_delta is None:
        return tool_calls

    tc_delta = tool_calls_delta[0]

    if len(tool_calls) == 0:
        tool_calls.append(tc_delta)
    else:
        # we need to either update latest tool_call or start a
        # new tool_call (i.e., multiple tools in this turn) and
        # accumulate that new tool_call with future delta chunks
        t = tool_calls[-1]
        if t.index != tc_delta.index:
            # the start of a new tool call, so append to our running tool_calls list
            tool_calls.append(tc_delta)
        else:
            # not the start of a new tool call, so update last item of tool_calls

            # validations to get passed by mypy
            assert t.function is not None
            assert tc_delta.function is not None
            assert t.function.arguments is not None
            assert t.function.name is not None
            assert t.id is not None

            t.function.arguments += tc_delta.function.arguments or ""
            t.function.name += tc_delta.function.name or ""
            t.id += tc_delta.id or ""
    return tool_calls
