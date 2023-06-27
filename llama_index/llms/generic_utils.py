from typing import Any, Callable, Sequence

from llama_index.llms.base import (
    ChatDeltaResponse,
    ChatMessage,
    ChatResponse,
    CompletionDeltaResponse,
    CompletionResponse,
    Message,
    StreamChatResponse,
    StreamCompletionResponse,
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

    string_messages.append("assistant: ")
    return "\n".join(string_messages)


def prompt_to_messages(prompt: str) -> Sequence[Message]:
    """Convert a string prompt to a sequence of messages."""
    return [ChatMessage(role="user", content=prompt)]


def completion_response_to_chat_response(
    completion_response: CompletionResponse,
) -> ChatResponse:
    """Convert a completion response to a chat response."""
    return ChatResponse(
        message=ChatMessage(
            role="assistant",
            content=completion_response.text,
            additional_kwargs=completion_response.additional_kwargs,
        ),
        raw=completion_response.raw,
    )


def stream_completion_response_to_chat_response(
    completion_response: StreamCompletionResponse,
) -> StreamChatResponse:
    """Convert a stream completion response to a stream chat response."""

    def gen() -> StreamChatResponse:
        for delta in completion_response:
            assert isinstance(delta, CompletionDeltaResponse)
            yield ChatDeltaResponse(
                message=ChatMessage(
                    role="assistant",
                    content=delta.text,
                    additional_kwargs=delta.additional_kwargs,
                ),
                delta=delta.delta,
                raw=delta.raw,
            )

    return gen()


def chat_response_to_completion_response(
    chat_response: ChatResponse,
) -> CompletionResponse:
    """Convert a chat response to a completion response."""
    return CompletionResponse(
        text=chat_response.message.content or "",
        additional_kwargs=chat_response.message.additional_kwargs,
        raw=chat_response.raw,
    )


def stream_chat_response_to_completion_response(
    chat_response: StreamChatResponse,
) -> StreamCompletionResponse:
    """Convert a stream chat response to a completion response."""

    def gen() -> StreamCompletionResponse:
        for delta in chat_response:
            assert isinstance(delta, ChatDeltaResponse)
            yield CompletionDeltaResponse(
                text=delta.message.content or "",
                additional_kwargs=delta.message.additional_kwargs,
                delta=delta.delta,
                raw=delta.raw,
            )

    return gen()


def completion_to_chat_decorator(
    func: Callable[..., CompletionResponse]
) -> Callable[..., ChatResponse]:
    """Convert a completion function to a chat function."""

    def wrapper(messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        # normalize input
        prompt = messages_to_prompt(messages)
        completion_response = func(prompt, **kwargs)
        # normalize output
        return completion_response_to_chat_response(completion_response)

    return wrapper


def stream_completion_to_chat_decorator(
    func: Callable[..., StreamCompletionResponse]
) -> Callable[..., StreamChatResponse]:
    """Convert a completion function to a chat function."""

    def wrapper(messages: Sequence[Message], **kwargs: Any) -> StreamChatResponse:
        # normalize input
        prompt = messages_to_prompt(messages)
        completion_response = func(prompt, **kwargs)
        # normalize output
        return stream_completion_response_to_chat_response(completion_response)

    return wrapper


def chat_to_completion_decorator(
    func: Callable[..., ChatResponse]
) -> Callable[..., CompletionResponse]:
    """Convert a chat function to a completion function."""

    def wrapper(prompt: str, **kwargs: Any) -> CompletionResponse:
        # normalize input
        messages = prompt_to_messages(prompt)
        chat_response = func(messages, **kwargs)
        # normalize output
        return chat_response_to_completion_response(chat_response)

    return wrapper


def stream_chat_to_completion_decorator(
    func: Callable[..., StreamChatResponse]
) -> Callable[..., StreamCompletionResponse]:
    """Convert a chat function to a completion function."""

    def wrapper(prompt: str, **kwargs: Any) -> StreamCompletionResponse:
        # normalize input
        messages = prompt_to_messages(prompt)
        chat_response = func(messages, **kwargs)
        # normalize output
        return stream_chat_response_to_completion_response(chat_response)

    return wrapper
