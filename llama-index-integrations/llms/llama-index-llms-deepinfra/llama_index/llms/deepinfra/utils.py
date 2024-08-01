import json

from typing import Union, Sequence, Dict, Any, Callable
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from asyncio import iscoroutinefunction

from requests.exceptions import Timeout, ConnectionError
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole


def maybe_decode_sse_data(data: bytes) -> Union[dict, None]:
    """
    Decode data from the streaming response.
    Checks whether the incoming data is an actual
    SSE data message.

    Args:
        data (bytes): The incoming data.

    Returns:
        Union[dict, None]: The decoded data or None.
    """
    if data and data.startswith(b"data: "):
        data = data.decode("utf-8").strip("data: ")
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None
    else:
        return None


def maybe_extract_from_json(data: dict, key: str = "text") -> Union[str, None]:
    """
    Extract text from a JSON response.

    Args:
        data (dict): The JSON response.

    Returns:
        Union[str, None]: The extracted text or None.
    """
    if "choices" in data:
        if len(data["choices"]) > 0 and key in data["choices"][0]:
            return data["choices"][0][key]
        else:
            return None
    else:
        return None


def chat_messages_to_list(messages: Sequence[ChatMessage]) -> Sequence[Dict[str, Any]]:
    """
    Convert a sequence of chat messages to a list of dictionaries.

    Args:
        messages (Sequence[ChatMessage]): A sequence of chat messages.

    Returns:
        Sequence[Dict[str, Any]]: A list of dictionaries.
    """
    chat_messages = []
    for message in messages:
        if message.role in [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.SYSTEM,
            MessageRole.TOOL,
        ]:
            chat_messages.append(
                {
                    "role": message.role,
                    "content": message.content,
                    **message.additional_kwargs,
                }
            )

    return chat_messages


def create_retry_decorator(retry_limit: int) -> Callable[[Any], Any]:
    """
    Create a retry decorator with the given retry limit.

    Args:
        retry_limit (int): The retry limit.

    Returns:
        Callable[[Any], Any]: The retry decorator.
    """
    initial_delay = 4
    max_delay = 10

    return retry(
        reraise=True,
        stop=stop_after_attempt(retry_limit),
        wait=wait_exponential(multiplier=1, min=initial_delay, max=max_delay),
        retry=(
            retry_if_exception_type(Timeout) | retry_if_exception_type(ConnectionError)
        ),
    )


def retry_request(
    request_func: Callable[..., Any], max_retries: int = 10, *args: Any, **kwargs: Any
) -> Any:
    """
    Retry a request function.

    Args:
        request_func (Callable[..., Any]): The request function.
        max_retries (int): The maximum number of retries.
        *args (Any): The positional arguments.
        **kwargs (Any): The keyword arguments.

    Returns:
        Any: The response.
    """
    retry_func = create_retry_decorator(max_retries)

    @retry_func
    def retry_func(request_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return request_func(*args, **kwargs)

    return retry_func(request_func, *args, **kwargs)


async def aretry_request(
    request_func: Callable[..., Any], max_retries: int = 10, *args: Any, **kwargs: Any
) -> Any:
    """
    Retry a request function asynchronously.

    Args:
        request_func (Callable[..., Any]): The request function.
        max_retries (int): The maximum number of retries.
        *args (Any): The positional arguments.
        **kwargs (Any): The keyword arguments.

    Returns:
        Any: The response.
    """
    retry_decorator = create_retry_decorator(max_retries)

    @retry_decorator
    async def retry_func(
        request_func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        if iscoroutinefunction(request_func):
            return await request_func(*args, **kwargs)
        else:
            return request_func(*args, **kwargs)

    return await retry_func(request_func, *args, **kwargs)


def force_single_tool_call(response: ChatResponse) -> None:
    """
    Force a single tool call in the response.
    Overrides the tool calls in the response message.
    """
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]
