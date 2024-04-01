from typing import Sequence
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
)
from text_generation.types import (
    Message,
)


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


def to_tgi_messages(messages: Sequence[ChatMessage]) -> Sequence[Message]:
    messages = []
    for m in messages:
        tool_calls = m.additional_kwargs.get("tool_calls")
        messages.append(
            Message(role=m.role.value, content=m.content, tool_calls=tool_calls)
        )

    return messages
