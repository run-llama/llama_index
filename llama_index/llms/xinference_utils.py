from typing_extensions import NotRequired, TypedDict

from llama_index.llms.base import ChatMessage

class ChatCompletionMessage(TypedDict):
    role: str
    content: str
    user: NotRequired[str]


def message_to_history(message: ChatMessage) -> ChatCompletionMessage:
    return ChatCompletionMessage(role=str(message.role), content=message.content)
