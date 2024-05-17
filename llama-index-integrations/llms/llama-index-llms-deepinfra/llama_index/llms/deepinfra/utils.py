import json

from typing import Union, Sequence, Dict, Any

from llama_index.core.base.llms.types import ChatMessage


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


def chat_messages_to_list(messages: Sequence[ChatMessage]) -> Sequence[Dict[str, Any]]:
    """
    Convert a sequence of chat messages to a list of dictionaries.

    Args:
        messages (Sequence[ChatMessage]): A sequence of chat messages.

    Returns:
        Sequence[Dict[str, Any]]: A list of dictionaries.
    """
    return [{"role": message.role, "content": message.content} for message in messages]
