from typing import Any
from llama_index.llms.base import (
    LLM,
    CompletionDeltaResponse,
    CompletionResponse,
    LLMMetadata,
    StreamCompletionResponse,
)
from llama_index.llms.custom import CustomLLM


class MockLLM(CustomLLM):
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(
            text=prompt,
        )

    def stream_complete(self, prompt: str, **kwargs: Any) -> StreamCompletionResponse:
        def gen() -> StreamCompletionResponse:
            for ch in prompt:
                yield CompletionDeltaResponse(
                    text=prompt,
                    delta=ch,
                )

        return gen()

    def metadata(self) -> LLMMetadata:
        return LLMMetadata()
