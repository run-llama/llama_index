from llama_index.llms.base import ChatResponseGen, CompletionResponseGen
from llama_index.types import TokenGen


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
