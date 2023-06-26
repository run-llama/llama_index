from typing import Any, Callable, Generator, Sequence

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

    string_messages.append("assistant: ")
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
            message=ChatMessage(
                role="assistant",
                content=completion_response.text,
                additional_kwargs=completion_response.additional_kwargs,
            ),
            raw=completion_response.raw,
        )
    elif isinstance(completion_response, Generator):

        def gen() -> Generator[ChatDeltaResponse, None, None]:
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

    else:
        return ValueError("Invalid completion response type.")


def chat_response_to_completion_response(
    chat_response: ChatResponseType,
) -> CompletionResponseType:
    """Convert a chat response to a completion response."""
    if isinstance(chat_response, ChatResponse):
        return CompletionResponse(
            text=chat_response.message.content or "",
            additional_kwargs=chat_response.message.additional_kwargs,
            raw=chat_response.raw,
        )
    elif isinstance(chat_response, Generator):

        def gen() -> Generator[CompletionDeltaResponse, None, None]:
            for delta in chat_response:
                assert isinstance(delta, ChatDeltaResponse)
                yield CompletionDeltaResponse(
                    text=delta.message.content or "",
                    additional_kwargs=delta.message.additional_kwargs,
                    delta=delta.delta,
                    raw=delta.raw,
                )

        return gen()

    else:
        return ValueError("Invalid chat response type.")


def completion_to_chat_decorator(
    func: Callable[..., CompletionResponse]
) -> Callable[..., ChatResponse]:
    """Convert a completion function to a chat function."""

    def wrapper(messages: Sequence[Message], **kwargs: Any) -> ChatResponseType:
        # normalize input
        prompt = messages_to_prompt(messages)
        completion_response = func(prompt, **kwargs)
        # normalize output
        return completion_response_to_chat_response(completion_response)

    return wrapper


def chat_to_completion_decorator(
    func: Callable[..., ChatResponse]
) -> Callable[..., CompletionResponse]:
    """Convert a chat function to a completion function."""

    def wrapper(prompt: str, **kwargs: Any) -> ChatResponse:
        # normalize input
        messages = prompt_to_messages(prompt)
        chat_response = func(messages, **kwargs)
        # normalize output
        return chat_response_to_completion_response(chat_response)

    return wrapper
