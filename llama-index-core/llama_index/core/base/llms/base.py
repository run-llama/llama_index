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
from llama_index.core.bridge.pydantic import Field, validator
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import BaseComponent


class BaseLLM(ChainableMixin, BaseComponent):
    """BaseLLM interface."""

    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    class Config:
        arbitrary_types_allowed = True

    @validator("callback_manager", pre=True)
    def _validate_callback_manager(cls, v: CallbackManager) -> CallbackManager:
        if v is None:
            return CallbackManager([])
        return v

    @property
    @abstractmethod
    def metadata(self) -> LLMMetadata:
        """LLM metadata.

        Returns:
        -------
        LLMMetadata
            LLM metadata containing various information about the LLM.
        """

    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat endpoint for LLM.

        Parameters
        ----------
        messages : Sequence[ChatMessage]
            Sequence of chat messages.
        kwargs : Any
            Additional keyword arguments to pass to the LLM.

        Returns:
        -------
        ChatResponse
            Chat response from the LLM.
        """

    @abstractmethod
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint for LLM.

        If the LLM is a chat model, the prompt is transformed into a single `user` message.

        Parameters
        ----------
        prompt : str
            Prompt to send to the LLM.
        formatted : bool, optional (default=False)
            Whether the prompt is already formatted for the LLM, by default False.
        kwargs : Any
            Additional keyword arguments to pass to the LLM.

        Returns:
        -------
        CompletionResponse
            Completion response from the LLM.
        """

    @abstractmethod
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Streaming chat endpoint for LLM.

        Parameters
        ----------
        messages : Sequence[ChatMessage]
            Sequence of chat messages.
        kwargs : Any
            Additional keyword arguments to pass to the LLM.

        Returns:
        -------
        ChatResponseGen
            A generator of ChatResponse objects, each containing a new token of the response.
        """

    @abstractmethod
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming completion endpoint for LLM.

        If the LLM is a chat model, the prompt is transformed into a single `user` message.

        Parameters
        ----------
        prompt : str
            Prompt to send to the LLM.
        formatted : bool, optional (default=False)
            Whether the prompt is already formatted for the LLM, by default False.
        kwargs : Any
            Additional keyword arguments to pass to the LLM.

        Returns:
        -------
        CompletionResponseGen
            A generator of CompletionResponse objects, each containing a new token of the response.
        """

    # ===== Async Endpoints =====
    @abstractmethod
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Async chat endpoint for LLM.

        Parameters
        ----------
        messages : Sequence[ChatMessage]
            Sequence of chat messages.
        kwargs : Any
            Additional keyword arguments to pass to the LLM.

        Returns:
        -------
        ChatResponse
            Chat response from the LLM.
        """

    @abstractmethod
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Async completion endpoint for LLM.

        If the LLM is a chat model, the prompt is transformed into a single `user` message.

        Parameters
        ----------
        prompt : str
            Prompt to send to the LLM.
        formatted : bool, optional (default=False)
            Whether the prompt is already formatted for the LLM, by default False.
        kwargs : Any
            Additional keyword arguments to pass to the LLM.

        Returns:
        -------
        CompletionResponse
            Completion response from the LLM.
        """

    @abstractmethod
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Async streaming chat endpoint for LLM.

        Parameters
        ----------
        messages : Sequence[ChatMessage]
            Sequence of chat messages.
        kwargs : Any
            Additional keyword arguments to pass to the LLM.

        Returns:
        -------
        ChatResponseAsyncGen
            An async generator of ChatResponse objects, each containing a new token of the response.
        """

    @abstractmethod
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Async streaming completion endpoint for LLM.

        If the LLM is a chat model, the prompt is transformed into a single `user` message.

        Parameters
        ----------
        prompt : str
            Prompt to send to the LLM.
        formatted : bool, optional (default=False)
            Whether the prompt is already formatted for the LLM, by default False.
        kwargs : Any
            Additional keyword arguments to pass to the LLM.

        Returns:
        -------
        CompletionResponseAsyncGen
            An async generator of CompletionResponse objects, each containing a new token of the response.
        """
