from llama_index.llm_predictor.base import StreamTokens
from llama_index.llms.base import StreamCompletionResponse


def stream_completion_response_to_tokens(
    completion_response: StreamCompletionResponse,
) -> StreamTokens:
    """Convert a stream completion response to a stream of tokens."""

    def gen() -> StreamTokens:
        for delta in completion_response:
            yield delta.delta

    return gen()
