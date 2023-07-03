from typing import Any, Callable, Sequence

from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    MessageRole,
)


def messages_to_history_str(messages: Sequence[ChatMessage]) -> str:
    """Convert messages to a history string."""
    string_messages = []
    for message in messages:
        role = message.role
        content = message.content
        string_message = f"{role}: {content}"

        addtional_kwargs = message.additional_kwargs
        if addtional_kwargs:
            string_message += f"\n{addtional_kwargs}"
        string_messages.append(string_message)
    return "\n".join(string_messages)


def messages_to_prompt(messages: Sequence[ChatMessage]) -> str:
    """Convert messages to a prompt string."""
    string_messages = []
    for message in messages:
        role = message.role
        content = message.content
        string_message = f"{role}: {content}"

        addtional_kwargs = message.additional_kwargs
        if addtional_kwargs:
            string_message += f"\n{addtional_kwargs}"
        string_messages.append(string_message)

    string_messages.append(f"{MessageRole.ASSISTANT}: ")
    return "\n".join(string_messages)


def prompt_to_messages(prompt: str) -> Sequence[ChatMessage]:
    """Convert a string prompt to a sequence of messages."""
    return [ChatMessage(role=MessageRole.USER, content=prompt)]


def completion_response_to_chat_response(
    completion_response: CompletionResponse,
) -> ChatResponse:
    """Convert a completion response to a chat response."""
    return ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=completion_response.text,
            additional_kwargs=completion_response.additional_kwargs,
        ),
        raw=completion_response.raw,
    )


def stream_completion_response_to_chat_response(
    completion_response_gen: CompletionResponseGen,
) -> ChatResponseGen:
    """Convert a stream completion response to a stream chat response."""

    def gen() -> ChatResponseGen:
        for response in completion_response_gen:
            yield ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.text,
                    additional_kwargs=response.additional_kwargs,
                ),
                delta=response.delta,
                raw=response.raw,
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
    chat_response_gen: ChatResponseGen,
) -> CompletionResponseGen:
    """Convert a stream chat response to a completion response."""

    def gen() -> CompletionResponseGen:
        for response in chat_response_gen:
            yield CompletionResponse(
                text=response.message.content or "",
                additional_kwargs=response.message.additional_kwargs,
                delta=response.delta,
                raw=response.raw,
            )

    return gen()


def completion_to_chat_decorator(
    func: Callable[..., CompletionResponse]
) -> Callable[..., ChatResponse]:
    """Convert a completion function to a chat function."""

    def wrapper(messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # normalize input
        prompt = messages_to_prompt(messages)
        completion_response = func(prompt, **kwargs)
        # normalize output
        return completion_response_to_chat_response(completion_response)

    return wrapper


def stream_completion_to_chat_decorator(
    func: Callable[..., CompletionResponseGen]
) -> Callable[..., ChatResponseGen]:
    """Convert a completion function to a chat function."""

    def wrapper(messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
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
    func: Callable[..., ChatResponseGen]
) -> Callable[..., CompletionResponseGen]:
    """Convert a chat function to a completion function."""

    def wrapper(prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # normalize input
        messages = prompt_to_messages(prompt)
        chat_response = func(messages, **kwargs)
        # normalize output
        return stream_chat_response_to_completion_response(chat_response)

    return wrapper
