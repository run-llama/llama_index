from typing import Any, Sequence
from llama_index.llms.base import LLM

from llama_index.llms.base import (
    ChatResponse,
    CompletionResponse,
    ChatMessage,
    StreamChatResponse,
    StreamCompletionResponse,
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

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        chat_fn = completion_to_chat_decorator(self.complete)
        return chat_fn(messages, **kwargs)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> StreamChatResponse:
        stream_chat_fn = stream_completion_to_chat_decorator(self.stream_complete)
        return stream_chat_fn(messages, **kwargs)

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        return self.chat(messages, **kwargs)

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> StreamChatResponse:
        return self.stream_chat(messages, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> StreamCompletionResponse:
        return self.stream_complete(prompt, **kwargs)
