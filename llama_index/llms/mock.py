from typing import Any
from llama_index.llms.base import (
    CompletionDeltaResponse,
    CompletionResponse,
    LLMMetadata,
    StreamCompletionResponse,
)
from llama_index.llms.custom import CustomLLM


class MockLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

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
