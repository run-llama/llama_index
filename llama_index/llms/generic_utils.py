from typing import Generator, Sequence

from llama_index.llms.base import (
    ChatDeltaResponse,
    ChatMessage,
    ChatResponse,
    ChatResponseType,
    CompletionDeltaResponse,
    CompletionResponse,
    CompletionResponseType,
    Message,
)


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


def completion_response_to_chat_response(
    completion_response: CompletionResponseType,
) -> ChatResponseType:
    """Convert a completion response to a chat response."""
    if isinstance(completion_response, CompletionResponse):
        return ChatResponse(
            message=ChatMessage(role="assistant", content=completion_response.text),
            raw=completion_response.raw,
        )
    elif isinstance(completion_response, Generator):

        def gen() -> Generator[ChatDeltaResponse, None, None]:
            for delta in completion_response:
                assert isinstance(delta, CompletionDeltaResponse)
                yield ChatDeltaResponse(
                    message=ChatMessage(role="assistant", content=delta.text),
                    delta=delta.delta,
                    raw=delta.raw,
                )

        return gen()

    else:
        return ValueError("Invalid completion response type.")


def chat_response_to_completion_response(
    chat_response: ChatResponseType,
) -> CompletionResponseType:
    """Convert a chat response to a completion response."""
    if isinstance(chat_response, ChatResponse):
        return CompletionResponse(
            text=chat_response.message.content,
            raw=chat_response.raw,
        )
    elif isinstance(chat_response, Generator):

        def gen() -> Generator[CompletionDeltaResponse, None, None]:
            for delta in chat_response:
                assert isinstance(delta, ChatDeltaResponse)
                yield CompletionDeltaResponse(
                    text=delta.message.content,
                    delta=delta.delta,
                    raw=delta.raw,
                )

        return gen()

    else:
        return ValueError("Invalid chat response type.")
