from typing import Any, Sequence

from llama_index.llms.base import (
    LLM,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.generic_utils import (
    completion_to_chat_decorator,
    stream_completion_to_chat_decorator,
)


class CustomLLM(LLM):
    """Simple abstract base class for custom LLMs.

    Subclasses must implement the `__init__`, `complete`,
        `stream_complete`, and `metadata` methods.
    """

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        chat_fn = completion_to_chat_decorator(self.complete)
        return chat_fn(messages, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        stream_chat_fn = stream_completion_to_chat_decorator(self.stream_complete)
        return stream_chat_fn(messages, **kwargs)

    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        return self.chat(messages, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        async def gen() -> ChatResponseAsyncGen:
            for message in self.stream_chat(messages, **kwargs):
                yield message

        # NOTE: convert generator to async generator
        return gen()

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        async def gen() -> CompletionResponseAsyncGen:
            for message in self.stream_complete(prompt, **kwargs):
                yield message

        # NOTE: convert generator to async generator
        return gen()

    @classmethod
    def class_name(cls) -> str:
        return "custom_llm"
