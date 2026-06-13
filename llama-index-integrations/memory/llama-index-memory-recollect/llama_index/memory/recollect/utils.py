from __future__ import annotations

from typing import Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole

DEFAULT_INTRO = "Below are relevant memories retrieved for this conversation:"
DEFAULT_OUTRO = "End of retrieved memories."


def convert_memory_to_system_message(
    response: list[dict[str, Any]],
    existing_system_message: ChatMessage | None = None,
) -> ChatMessage:
    lines = [str(item.get("memory", "")) for item in response]
    formatted = "\n\n" + DEFAULT_INTRO + "\n"
    for line in lines:
        if line:
            formatted += f"\n- {line}\n"
    formatted += "\n" + DEFAULT_OUTRO
    if existing_system_message is not None and existing_system_message.content:
        base = existing_system_message.content.split(DEFAULT_INTRO)[0]
        formatted = base + formatted
    return ChatMessage(content=formatted, role=MessageRole.SYSTEM)


def convert_chat_history_to_dict(messages: list[ChatMessage]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for message in messages:
        if message.role in (MessageRole.USER, MessageRole.ASSISTANT) and message.content:
            out.append({"role": message.role.value, "content": message.content})
    return out


def convert_messages_to_string(
    messages: list[ChatMessage], input: str | None = None, limit: int = 5
) -> str:
    recent = messages[-limit:]
    formatted = [f"{msg.role.value}: {msg.content}" for msg in recent]
    result = "\n".join(formatted)
    if input:
        result += f"\nuser: {input}"
    return result