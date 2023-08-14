from typing import Any
from llama_index.llms.base import (
    ChatMessage,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.custom import CustomLLM


class TestLLM(CustomLLM):
    def __init__(self) -> None:
        super().__init__(callback_manager=None)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(
            text="test output",
            additional_kwargs={
                "prompt": prompt,
            },
        )

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        def gen() -> CompletionResponseGen:
            text = "test output"
            text_so_far = ""
            for ch in text:
                text_so_far += ch
                yield CompletionResponse(
                    text=text_so_far,
                    delta=ch,
                    additional_kwargs={
                        "prompt": prompt,
                    },
                )

        return gen()


def test_basic() -> None:
    llm = TestLLM()

    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    llm.complete(prompt)
    llm.chat([message])


def test_streaming() -> None:
    llm = TestLLM()

    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    llm.stream_complete(prompt)
    llm.stream_chat([message])
