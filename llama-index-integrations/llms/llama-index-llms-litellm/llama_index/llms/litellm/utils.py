import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

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
from litellm import completion
from litellm.utils import Message
from litellm.main import ModelResponse, CustomStreamWrapper
from litellm.types.utils import ChatCompletionDeltaToolCall
from llama_index.core.base.llms.types import (
    TextBlock,
    ImageBlock,
    AudioBlock,
)


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
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    def _completion_with_retry(
        **kwargs: Any,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        return completion(**kwargs)

    return _completion_with_retry(**kwargs)


async def acompletion_with_retry(
    is_chat_model: bool, max_retries: int, **kwargs: Any
) -> Any:
    from litellm import acompletion

    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    async def _completion_with_retry(
        **kwargs: Any,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
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


def is_function_calling_model(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    import litellm

    return litellm.supports_function_calling(
        model, custom_llm_provider=custom_llm_provider
    )


def get_completion_endpoint(is_chat_model: bool) -> CompletionClientType:
    from litellm import completion

    return completion


def to_openailike_message_dict(message: ChatMessage) -> dict:
    """Convert a ChatMessage to an OpenAI-like message dict."""
    content = []
    content_txt = ""
    for block in message.blocks:
        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
            content_txt += block.text
        elif isinstance(block, ImageBlock):
            if block.url:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": str(block.url),
                            "detail": block.detail or "auto",
                        },
                    }
                )
            else:
                img_bytes = block.resolve_image(as_base64=True).read()
                img_str = img_bytes.decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.image_mimetype};base64,{img_str}",
                            "detail": block.detail or "auto",
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

    # use content as text instead of blocks if all blocks are text. Safer for backwards compatibility with older openai-like APIs
    message_dict = {
        "role": message.role.value,
        "content": (
            content_txt
            if all(isinstance(block, TextBlock) for block in message.blocks)
            else content
        ),
    }

    message_dict.update(message.additional_kwargs)

    return message_dict


def to_openai_message_dicts(messages: Sequence[ChatMessage]) -> List[dict]:
    """Convert generic messages to OpenAI message dicts."""
    return [to_openailike_message_dict(message) for message in messages]


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
    tool_calls = message.get("tool_calls")
    additional_kwargs = {}
    if tool_calls:
        additional_kwargs["tool_calls"] = tool_calls

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


def validate_litellm_api_key(
    api_key: Optional[str] = None, api_type: Optional[str] = None
) -> None:
    import litellm

    api_key = litellm.validate_environment()
    if api_key is None:
        raise ValueError(MISSING_API_KEY_ERROR_MESSAGE)


def update_tool_calls(
    tool_calls: List[dict],
    tool_call_deltas: Optional[List[ChatCompletionDeltaToolCall]],
) -> List[dict]:
    """
    Update the list of tool calls with deltas.

    Args:
        tool_calls: The current list of tool calls
        tool_call_deltas: A list of deltas to update tool_calls with

    Returns:
        List[dict]: The updated tool calls
    """
    if not tool_call_deltas:
        return tool_calls

    for tool_call_delta in tool_call_deltas:
        # Convert ChatCompletionDeltaToolCall to dict
        delta_dict = {}
        if tool_call_delta.id is not None:
            delta_dict["id"] = tool_call_delta.id
        if tool_call_delta.type is not None:
            delta_dict["type"] = tool_call_delta.type
        if hasattr(tool_call_delta, "index"):
            delta_dict["index"] = tool_call_delta.index

        if (
            hasattr(tool_call_delta, "function")
            and tool_call_delta.function is not None
        ):
            delta_dict["function"] = {}
            if (
                hasattr(tool_call_delta.function, "name")
                and tool_call_delta.function.name is not None
            ):
                delta_dict["function"]["name"] = tool_call_delta.function.name
            if (
                hasattr(tool_call_delta.function, "arguments")
                and tool_call_delta.function.arguments is not None
            ):
                delta_dict["function"]["arguments"] = tool_call_delta.function.arguments

        if len(tool_calls) == 0:
            # First tool call
            tool_calls.append(delta_dict)
        else:
            # Try to find an existing tool call to update
            found_match = False
            for existing_tool in tool_calls:
                # Check by index if available
                index_match = False
                if "index" in delta_dict and "index" in existing_tool:
                    index_match = delta_dict["index"] == existing_tool["index"]

                # Check by id if available
                id_match = False
                if "id" in delta_dict and "id" in existing_tool:
                    id_match = delta_dict["id"] == existing_tool["id"]

                if index_match or id_match:
                    found_match = True
                    # Update existing tool call
                    if "function" in delta_dict:
                        if "function" not in existing_tool:
                            existing_tool["function"] = {}

                        # Update function name if present
                        if "name" in delta_dict["function"]:
                            if "name" not in existing_tool["function"]:
                                existing_tool["function"]["name"] = ""
                            existing_tool["function"]["name"] += delta_dict[
                                "function"
                            ].get("name", "")

                        # Update arguments if present
                        if "arguments" in delta_dict["function"]:
                            if "arguments" not in existing_tool["function"]:
                                existing_tool["function"]["arguments"] = ""
                            existing_tool["function"]["arguments"] += delta_dict[
                                "function"
                            ].get("arguments", "")

                    # Update ID if present
                    if "id" in delta_dict:
                        if "id" not in existing_tool:
                            existing_tool["id"] = ""
                        existing_tool["id"] += delta_dict.get("id", "")

                    # Update type if present
                    if "type" in delta_dict:
                        existing_tool["type"] = delta_dict["type"]

                    # Update index if present
                    if "index" in delta_dict:
                        existing_tool["index"] = delta_dict["index"]

                    break

            # If no match was found, add as a new tool call
            if not found_match and ("id" in delta_dict or "index" in delta_dict):
                tool_calls.append(delta_dict)

    return tool_calls
