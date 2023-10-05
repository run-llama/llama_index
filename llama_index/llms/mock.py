from typing import Any, Optional

from llama_index.callbacks import CallbackManager
from llama_index.llms.base import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    llm_completion_callback,
)
from llama_index.llms.custom import CustomLLM


class MockLLM(CustomLLM):
    max_tokens: Optional[int]

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__(max_tokens=max_tokens, callback_manager=callback_manager)

    @classmethod
    def class_name(cls) -> str:
        return "MockLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(num_output=self.max_tokens or -1)

    def _generate_text(self, length: int) -> str:
        return " ".join(["text" for _ in range(length)])

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response_text = (
            self._generate_text(self.max_tokens) if self.max_tokens else prompt
        )

        return CompletionResponse(
            text=response_text,
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        def gen_prompt() -> CompletionResponseGen:
            for ch in prompt:
                yield CompletionResponse(
                    text=prompt,
                    delta=ch,
                )

        def gen_response(max_tokens: int) -> CompletionResponseGen:
            for i in range(max_tokens):
                response_text = self._generate_text(i)
                yield CompletionResponse(
                    text=response_text,
                    delta="text ",
                )

        return gen_response(self.max_tokens) if self.max_tokens else gen_prompt()
