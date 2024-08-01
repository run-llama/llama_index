"""DashScope api utils."""

from http import HTTPStatus
from typing import Any, Dict, List, Sequence

from llama_index.legacy.core.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)


def dashscope_response_to_completion_response(
    response: Any, stream: bool = False
) -> CompletionResponse:
    if response["status_code"] == HTTPStatus.OK:
        content = response["output"]["choices"][0]["message"]["content"]
        if not content:
            content = ""
        return CompletionResponse(text=content, raw=response)
    else:
        return CompletionResponse(text="", raw=response)


def dashscope_response_to_chat_response(
    response: Any,
) -> ChatResponse:
    if response["status_code"] == HTTPStatus.OK:
        content = response["output"]["choices"][0]["message"]["content"]
        if not content:
            content = ""
        role = response["output"]["choices"][0]["message"]["role"]
        return ChatResponse(
            message=ChatMessage(role=role, content=content), raw=response
        )
    else:
        return ChatResponse(message=ChatMessage(), raw=response)


def chat_message_to_dashscope_messages(
    chat_messages: Sequence[ChatMessage],
) -> List[Dict]:
    messages = []
    for msg in chat_messages:
        messages.append({"role": msg.role.value, "content": msg.content})
    return messages
