from llama_index.llms.base import (
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from llama_index.types import TokenAsyncGen, TokenGen


def stream_completion_response_to_tokens(
    completion_response_gen: CompletionResponseGen,
) -> TokenGen:
    """Convert a stream completion response to a stream of tokens."""

    def gen() -> TokenGen:
        for response in completion_response_gen:
            yield response.delta or ""

    return gen()


def stream_chat_response_to_tokens(
    chat_response_gen: ChatResponseGen,
) -> TokenGen:
    """Convert a stream completion response to a stream of tokens."""

    def gen() -> TokenGen:
        for response in chat_response_gen:
            yield response.delta or ""

    return gen()


async def astream_completion_response_to_tokens(
    completion_response_gen: CompletionResponseAsyncGen,
) -> TokenAsyncGen:
    """Convert a stream completion response to a stream of tokens."""

    async def gen() -> TokenAsyncGen:
        async for response in completion_response_gen:
            yield response.delta or ""

    return gen()


async def astream_chat_response_to_tokens(
    chat_response_gen: ChatResponseAsyncGen,
) -> TokenAsyncGen:
    """Convert a stream completion response to a stream of tokens."""

    async def gen() -> TokenAsyncGen:
        async for response in chat_response_gen:
            yield response.delta or ""

    return gen()
