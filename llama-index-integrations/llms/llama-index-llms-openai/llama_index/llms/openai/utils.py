import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import openai
from deprecated import deprecated
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.completion_choice import Logprobs
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

from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    LogProb,
    MessageRole,
    TextBlock,
    AudioBlock,
)
from llama_index.core.bridge.pydantic import BaseModel

DEFAULT_OPENAI_API_TYPE = "open_ai"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_OPENAI_API_VERSION = ""

O1_MODELS: Dict[str, int] = {
    "o1": 200000,
    "o1-2024-12-17": 200000,
    "o1-preview": 128000,
    "o1-preview-2024-09-12": 128000,
    "o1-mini": 128000,
    "o1-mini-2024-09-12": 128000,
    "o3-mini": 200000,
    "o3-mini-2025-01-31": 200000,
}

O1_MODELS_WITHOUT_FUNCTION_CALLING = {
    "o1-preview",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
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
    "gpt-4o-audio-preview": 128000,
    "gpt-4o-audio-preview-2024-12-17": 128000,
    "gpt-4o-audio-preview-2024-10-01": 128000,
    "gpt-4o-mini-audio-preview": 128000,
    "gpt-4o-mini-audio-preview-2024-12-17": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-2024-11-20": 128000,
    # Intended for research and evaluation
    "chatgpt-4o-latest": 128000,
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
    "gpt-4o-mini": 128000,
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
    """
    Calculate the maximum number of tokens possible to generate for a model.

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
    # checking whether the model is fine-tuned or not.
    # fine-tuned model names these days look like:
    # ft:gpt-3.5-turbo:acemeco:suffix:abc123
    if model.startswith("ft-"):  # legacy fine-tuning
        model = model.split(":")[0]
    elif model.startswith("ft:"):
        model = model.split(":")[1]

    is_chat_model_ = is_chat_model(model)
    is_old = "0314" in model or "0301" in model
    is_o1_beta = model in O1_MODELS_WITHOUT_FUNCTION_CALLING

    return is_chat_model_ and not is_old and not is_o1_beta


def to_openai_message_dict(
    message: ChatMessage,
    drop_none: bool = False,
    model: Optional[str] = None,
) -> ChatCompletionMessageParam:
    """Convert a ChatMessage to an OpenAI message dict."""
    content = []
    content_txt = ""
    reference_audio_id = None
    for block in message.blocks:
        if message.role == MessageRole.ASSISTANT:
            reference_audio_id = message.additional_kwargs.get(
                "reference_audio_id", None
            )
            # if reference audio id is provided, we don't need to send the audio
            if reference_audio_id:
                continue

        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
            content_txt += block.text
        elif isinstance(block, ImageBlock):
            if block.url:
                content.append(
                    {"type": "image_url", "image_url": {"url": str(block.url)}}
                )
            else:
                img_bytes = block.resolve_image(as_base64=True).read()
                img_str = img_bytes.decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.image_mimetype};base64,{img_str}",
                            "detail": block.detail or "low",
                        },
                    }
                )
        elif isinstance(block, AudioBlock):
            audio_bytes = block.resolve_audio(as_base64=True).read()
            audio_str = audio_bytes.decode("utf-8")
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_str,
                        "format": block.format,
                    },
                }
            )
        else:
            msg = f"Unsupported content block type: {type(block).__name__}"
            raise ValueError(msg)

    # NOTE: Sending a null value (None) for Tool Message to OpenAI will cause error
    # It's only Allowed to send None if it's an Assistant Message
    # Reference: https://platform.openai.com/docs/api-reference/chat/create
    content_txt = (
        None
        if content_txt == "" and message.role == MessageRole.ASSISTANT
        else content_txt
    )

    # If reference audio id is provided, we don't need to send the audio
    # NOTE: this is only a thing for assistant messages
    if reference_audio_id:
        message_dict = {
            "role": message.role.value,
            "audio": {"id": reference_audio_id},
        }
    else:
        # NOTE: Despite what the openai docs say, if the role is ASSISTANT, SYSTEM
        # or TOOL, 'content' cannot be a list and must be string instead.
        # Furthermore, if all blocks are text blocks, we can use the content_txt
        # as the content. This will avoid breaking openai-like APIs.
        message_dict = {
            "role": message.role.value,
            "content": (
                content_txt
                if message.role.value in ("assistant", "tool", "system")
                or all(isinstance(block, TextBlock) for block in message.blocks)
                else content
            ),
        }

    # TODO: O1 models do not support system prompts
    if (
        model is not None
        and model in O1_MODELS
        and model not in O1_MODELS_WITHOUT_FUNCTION_CALLING
    ):
        if message_dict["role"] == "system":
            message_dict["role"] = "developer"

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
        to_openai_message_dict(
            message,
            drop_none=drop_none,
            model=model,
        )
        for message in messages
    ]


def from_openai_message(
    openai_message: ChatCompletionMessage, modalities: List[str]
) -> ChatMessage:
    """Convert openai message dict to generic message."""
    role = openai_message.role
    # NOTE: Azure OpenAI returns function calling messages without a content key
    if "text" in modalities and openai_message.content:
        blocks = [TextBlock(text=openai_message.content or "")]
    else:
        blocks = []

    additional_kwargs: Dict[str, Any] = {}
    if openai_message.tool_calls:
        tool_calls: List[ChatCompletionMessageToolCall] = openai_message.tool_calls
        additional_kwargs.update(tool_calls=tool_calls)

    if openai_message.audio and "audio" in modalities:
        reference_audio_id = openai_message.audio.id
        audio_data = openai_message.audio.data
        additional_kwargs["reference_audio_id"] = reference_audio_id
        blocks.append(AudioBlock(audio=audio_data, format="mp3"))

    return ChatMessage(role=role, blocks=blocks, additional_kwargs=additional_kwargs)


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
        except Exception:
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
    openai_completion_logprob: Dict[str, float],
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


def from_openai_messages(
    openai_messages: Sequence[ChatCompletionMessage], modalities: List[str]
) -> List[ChatMessage]:
    """Convert openai message dicts to generic messages."""
    return [from_openai_message(message, modalities) for message in openai_messages]


def from_openai_message_dict(message_dict: dict) -> ChatMessage:
    """Convert openai message dict to generic message."""
    role = message_dict["role"]
    # NOTE: Azure OpenAI returns function calling messages without a content key
    content = message_dict.get("content")
    blocks = []
    if isinstance(content, list):
        for elem in content:
            t = elem.get("type")
            if t == "text":
                blocks.append(TextBlock(text=elem.get("text")))
            elif t == "image_url":
                img = elem["image_url"]["url"]
                detail = elem["image_url"]["detail"]
                if img.startswith("data:"):
                    blocks.append(ImageBlock(image=img, detail=detail))
                else:
                    blocks.append(ImageBlock(url=img, detail=detail))
            else:
                msg = f"Unsupported message type: {t}"
                raise ValueError(msg)
        content = None

    additional_kwargs = message_dict.copy()
    additional_kwargs.pop("role")
    additional_kwargs.pop("content", None)

    return ChatMessage(
        role=role, content=content, additional_kwargs=additional_kwargs, blocks=blocks
    )


def from_openai_message_dicts(message_dicts: Sequence[dict]) -> List[ChatMessage]:
    """Convert openai message dicts to generic messages."""
    return [from_openai_message_dict(message_dict) for message_dict in message_dicts]


@deprecated("Deprecated in favor of `to_openai_tool`, which should be used instead.")
def to_openai_function(pydantic_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    Deprecated in favor of `to_openai_tool`.

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
    """
    "Resolve OpenAI credentials.

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


def resolve_tool_choice(tool_choice: Union[str, dict] = "auto") -> Union[str, dict]:
    """
    Resolve tool choice.

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
