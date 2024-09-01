from abc import abstractmethod
from typing import (
    Any,
    Sequence,
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.base.query_pipeline.query import (
    ChainableMixin,
)
from llama_index.core.bridge.pydantic import Field, model_validator, ConfigDict
from llama_index.core.callbacks import CallbackManager
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.schema import BaseComponent


class BaseLLM(ChainableMixin, BaseComponent, DispatcherSpanMixin):
    """BaseLLM interface."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    callback_manager: CallbackManager = Field(
        default_factory=lambda: CallbackManager([]), exclude=True
    )

    @model_validator(mode="after")
    def check_callback_manager(self) -> "BaseLLM":
        if self.callback_manager is None:
            self.callback_manager = CallbackManager([])
        return self

    @property
    @abstractmethod
    def metadata(self) -> LLMMetadata:
        """LLM metadata.

        Returns:
            LLMMetadata: LLM metadata containing various information about the LLM.
        """

    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat endpoint for LLM.

        Args:
            messages (Sequence[ChatMessage]):
                Sequence of chat messages.
            kwargs (Any):
                Additional keyword arguments to pass to the LLM.

        Returns:
            ChatResponse: Chat response from the LLM.

        Examples:
            ```python
            from llama_index.core.llms import ChatMessage

            response = llm.chat([ChatMessage(role="user", content="Hello")])
            print(response.content)
            ```
        """

    @abstractmethod
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint for LLM.

        If the LLM is a chat model, the prompt is transformed into a single `user` message.

        Args:
            prompt (str):
                Prompt to send to the LLM.
            formatted (bool, optional):
                Whether the prompt is already formatted for the LLM, by default False.
            kwargs (Any):
                Additional keyword arguments to pass to the LLM.

        Returns:
            CompletionResponse: Completion response from the LLM.

        Examples:
            ```python
            response = llm.complete("your prompt")
            print(response.text)
            ```
        """

    @abstractmethod
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Streaming chat endpoint for LLM.

        Args:
            messages (Sequence[ChatMessage]):
                Sequence of chat messages.
            kwargs (Any):
                Additional keyword arguments to pass to the LLM.

        Yields:
            ChatResponse:
                A generator of ChatResponse objects, each containing a new token of the response.

        Examples:
            ```python
            from llama_index.core.llms import ChatMessage

            gen = llm.stream_chat([ChatMessage(role="user", content="Hello")])
            for response in gen:
                print(response.delta, end="", flush=True)
            ```
        """

    @abstractmethod
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming completion endpoint for LLM.

        If the LLM is a chat model, the prompt is transformed into a single `user` message.

        Args:
            prompt (str):
                Prompt to send to the LLM.
            formatted (bool, optional):
                Whether the prompt is already formatted for the LLM, by default False.
            kwargs (Any):
                Additional keyword arguments to pass to the LLM.

        Yields:
            CompletionResponse:
                A generator of CompletionResponse objects, each containing a new token of the response.

        Examples:
            ```python
            gen = llm.stream_complete("your prompt")
            for response in gen:
                print(response.text, end="", flush=True)
            ```
        """

    # ===== Async Endpoints =====
    @abstractmethod
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Async chat endpoint for LLM.

        Args:
            messages (Sequence[ChatMessage]):
                Sequence of chat messages.
            kwargs (Any):
                Additional keyword arguments to pass to the LLM.

        Returns:
            ChatResponse: Chat response from the LLM.

        Examples:
            ```python
            from llama_index.core.llms import ChatMessage

            response = await llm.achat([ChatMessage(role="user", content="Hello")])
            print(response.content)
            ```
        """

    @abstractmethod
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Async completion endpoint for LLM.

        If the LLM is a chat model, the prompt is transformed into a single `user` message.

        Args:
            prompt (str):
                Prompt to send to the LLM.
            formatted (bool, optional):
                Whether the prompt is already formatted for the LLM, by default False.
            kwargs (Any):
                Additional keyword arguments to pass to the LLM.

        Returns:
            CompletionResponse: Completion response from the LLM.

        Examples:
            ```python
            response = await llm.acomplete("your prompt")
            print(response.text)
            ```
        """

    @abstractmethod
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Async streaming chat endpoint for LLM.

        Args:
            messages (Sequence[ChatMessage]):
                Sequence of chat messages.
            kwargs (Any):
                Additional keyword arguments to pass to the LLM.

        Yields:
            ChatResponse:
                An async generator of ChatResponse objects, each containing a new token of the response.

        Examples:
            ```python
            from llama_index.core.llms import ChatMessage

            gen = await llm.astream_chat([ChatMessage(role="user", content="Hello")])
            async for response in gen:
                print(response.delta, end="", flush=True)
            ```
        """

    @abstractmethod
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Async streaming completion endpoint for LLM.

        If the LLM is a chat model, the prompt is transformed into a single `user` message.

        Args:
            prompt (str):
                Prompt to send to the LLM.
            formatted (bool, optional):
                Whether the prompt is already formatted for the LLM, by default False.
            kwargs (Any):
                Additional keyword arguments to pass to the LLM.

        Yields:
            CompletionResponse:
                An async generator of CompletionResponse objects, each containing a new token of the response.

        Examples:
            ```python
            gen = await llm.astream_complete("your prompt")
            async for response in gen:
                print(response.text, end="", flush=True)
            ```
        """
