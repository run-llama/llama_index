from typing import Any, Sequence

from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
)
from llama_index.llms.base import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.llms.llm import LLM


class CustomLLM(LLM):
    """Simple abstract base class for custom LLMs.

    Subclasses must implement the `__init__`, `_complete`,
        `_stream_complete`, and `metadata` methods.
    """

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response_gen = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response_gen)

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
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self.complete(prompt, formatted=formatted, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        async def gen() -> CompletionResponseAsyncGen:
            for message in self.stream_complete(prompt, formatted=formatted, **kwargs):
                yield message

        # NOTE: convert generator to async generator
        return gen()

    @classmethod
    def class_name(cls) -> str:
        return "custom_llm"
