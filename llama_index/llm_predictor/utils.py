from llama_index.llms.base import CompletionResponseGen
from llama_index.types import TokenGen


def stream_completion_response_to_tokens(
    completion_response: CompletionResponseGen,
) -> TokenGen:
    """Convert a stream completion response to a stream of tokens."""

    def gen() -> TokenGen:
        for delta in completion_response:
            yield delta.delta

    return gen()
