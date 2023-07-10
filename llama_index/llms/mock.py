from typing import Any

from llama_index.llms.base import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.llms.custom import CustomLLM


class MockLLM(CustomLLM):
    def __init__(self, max_tokens: int = 256):
        self.max_tokens = max_tokens

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    def _generate_text(self, length: int) -> str:
        return " ".join(["text" for _ in range(length)])

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(
            text=self._generate_text(self.max_tokens),
        )

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        def gen() -> CompletionResponseGen:
            for i in range(self.max_tokens):
                yield CompletionResponse(
                    text=self._generate_text(i),
                    delta="text ",
                )

        return gen()
