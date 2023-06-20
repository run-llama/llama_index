


from typing import Sequence

from llama_index.llms.base import ChatMessage, Message


def messages_to_prompt(messages: Sequence[Message]) -> str:
    """Convert messages to a string prompt."""
    string_messages = []
    for message in messages:
        if isinstance(message, ChatMessage):
            role = message.role
        else:
            role = "unknown"

        content = message.content
        string_message = f"{role}: {content}"

        addtional_kwargs = message.additional_kwargs
        if addtional_kwargs:
            string_message += f"\n{addtional_kwargs}"
        string_messages.append(string_message)
    return "\n".join(string_messages)


def prompt_to_messages(prompt: str) -> Sequence[Message]:
    """Convert a string prompt to a sequence of messages."""
    return [ChatMessage(role="user", content=prompt)]