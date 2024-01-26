import os
from typing import Any, Awaitable, Callable, List, Optional, Sequence

from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)


def messages_to_history_str(messages: Sequence[ChatMessage]) -> str:
    """Convert messages to a history string."""
    string_messages = []
    for message in messages:
        role = message.role
        content = message.content
        string_message = f"{role.value}: {content}"

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
        string_message = f"{role.value}: {content}"

        addtional_kwargs = message.additional_kwargs
        if addtional_kwargs:
            string_message += f"\n{addtional_kwargs}"
        string_messages.append(string_message)

    string_messages.append(f"{MessageRole.ASSISTANT.value}: ")
    return "\n".join(string_messages)


def prompt_to_messages(prompt: str) -> List[ChatMessage]:
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


def astream_completion_response_to_chat_response(
    completion_response_gen: CompletionResponseAsyncGen,
) -> ChatResponseAsyncGen:
    """Convert an async stream completion to an async stream chat response."""

    async def gen() -> ChatResponseAsyncGen:
        async for response in completion_response_gen:
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


# ===== Async =====


def acompletion_to_chat_decorator(
    func: Callable[..., Awaitable[CompletionResponse]]
) -> Callable[..., Awaitable[ChatResponse]]:
    """Convert a completion function to a chat function."""

    async def wrapper(messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # normalize input
        prompt = messages_to_prompt(messages)
        completion_response = await func(prompt, **kwargs)
        # normalize output
        return completion_response_to_chat_response(completion_response)

    return wrapper


def achat_to_completion_decorator(
    func: Callable[..., Awaitable[ChatResponse]]
) -> Callable[..., Awaitable[CompletionResponse]]:
    """Convert a chat function to a completion function."""

    async def wrapper(prompt: str, **kwargs: Any) -> CompletionResponse:
        # normalize input
        messages = prompt_to_messages(prompt)
        chat_response = await func(messages, **kwargs)
        # normalize output
        return chat_response_to_completion_response(chat_response)

    return wrapper


def astream_completion_to_chat_decorator(
    func: Callable[..., Awaitable[CompletionResponseAsyncGen]]
) -> Callable[..., Awaitable[ChatResponseAsyncGen]]:
    """Convert a completion function to a chat function."""

    async def wrapper(
        messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        # normalize input
        prompt = messages_to_prompt(messages)
        completion_response = await func(prompt, **kwargs)
        # normalize output
        return astream_completion_response_to_chat_response(completion_response)

    return wrapper


def astream_chat_to_completion_decorator(
    func: Callable[..., Awaitable[ChatResponseAsyncGen]]
) -> Callable[..., Awaitable[CompletionResponseAsyncGen]]:
    """Convert a chat function to a completion function."""

    async def wrapper(prompt: str, **kwargs: Any) -> CompletionResponseAsyncGen:
        # normalize input
        messages = prompt_to_messages(prompt)
        chat_response = await func(messages, **kwargs)
        # normalize output
        return astream_chat_response_to_completion_response(chat_response)

    return wrapper


def async_stream_completion_response_to_chat_response(
    completion_response_gen: CompletionResponseAsyncGen,
) -> ChatResponseAsyncGen:
    """Convert a stream completion response to a stream chat response."""

    async def gen() -> ChatResponseAsyncGen:
        async for response in completion_response_gen:
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


def astream_chat_response_to_completion_response(
    chat_response_gen: ChatResponseAsyncGen,
) -> CompletionResponseAsyncGen:
    """Convert a stream chat response to a completion response."""

    async def gen() -> CompletionResponseAsyncGen:
        async for response in chat_response_gen:
            yield CompletionResponse(
                text=response.message.content or "",
                additional_kwargs=response.message.additional_kwargs,
                delta=response.delta,
                raw=response.raw,
            )

    return gen()


def get_from_param_or_env(
    key: str,
    param: Optional[str] = None,
    env_key: Optional[str] = None,
    default: Optional[str] = None,
) -> str:
    """Get a value from a param or an environment variable."""
    if param is not None:
        return param
    elif env_key and env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )
