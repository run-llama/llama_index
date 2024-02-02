import asyncio
from abc import abstractmethod
from contextlib import contextmanager
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Sequence,
    cast,
)

from llama_index.bridge.pydantic import Field, validator
from llama_index.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.query_pipeline.query_component import (
    ChainableMixin,
)
from llama_index.schema import BaseComponent


def llm_chat_callback() -> Callable:
    def wrap(f: Callable) -> Callable:
        @contextmanager
        def wrapper_logic(_self: Any) -> Generator[CallbackManager, None, None]:
            callback_manager = getattr(_self, "callback_manager", None)
            if not isinstance(callback_manager, CallbackManager):
                raise ValueError(
                    "Cannot use llm_chat_callback on an instance "
                    "without a callback_manager attribute."
                )

            yield callback_manager

        async def wrapped_async_llm_chat(
            _self: Any, messages: Sequence[ChatMessage], **kwargs: Any
        ) -> Any:
            with wrapper_logic(_self) as callback_manager:
                event_id = callback_manager.on_event_start(
                    CBEventType.LLM,
                    payload={
                        EventPayload.MESSAGES: messages,
                        EventPayload.ADDITIONAL_KWARGS: kwargs,
                        EventPayload.SERIALIZED: _self.to_dict(),
                    },
                )

                f_return_val = await f(_self, messages, **kwargs)
                if isinstance(f_return_val, AsyncGenerator):
                    # intercept the generator and add a callback to the end
                    async def wrapped_gen() -> ChatResponseAsyncGen:
                        last_response = None
                        async for x in f_return_val:
                            yield cast(ChatResponse, x)
                            last_response = x

                        callback_manager.on_event_end(
                            CBEventType.LLM,
                            payload={
                                EventPayload.MESSAGES: messages,
                                EventPayload.RESPONSE: last_response,
                            },
                            event_id=event_id,
                        )

                    return wrapped_gen()
                else:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={
                            EventPayload.MESSAGES: messages,
                            EventPayload.RESPONSE: f_return_val,
                        },
                        event_id=event_id,
                    )

            return f_return_val

        def wrapped_llm_chat(
            _self: Any, messages: Sequence[ChatMessage], **kwargs: Any
        ) -> Any:
            with wrapper_logic(_self) as callback_manager:
                event_id = callback_manager.on_event_start(
                    CBEventType.LLM,
                    payload={
                        EventPayload.MESSAGES: messages,
                        EventPayload.ADDITIONAL_KWARGS: kwargs,
                        EventPayload.SERIALIZED: _self.to_dict(),
                    },
                )
                f_return_val = f(_self, messages, **kwargs)

                if isinstance(f_return_val, Generator):
                    # intercept the generator and add a callback to the end
                    def wrapped_gen() -> ChatResponseGen:
                        last_response = None
                        for x in f_return_val:
                            yield cast(ChatResponse, x)
                            last_response = x

                        callback_manager.on_event_end(
                            CBEventType.LLM,
                            payload={
                                EventPayload.MESSAGES: messages,
                                EventPayload.RESPONSE: last_response,
                            },
                            event_id=event_id,
                        )

                    return wrapped_gen()
                else:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={
                            EventPayload.MESSAGES: messages,
                            EventPayload.RESPONSE: f_return_val,
                        },
                        event_id=event_id,
                    )

            return f_return_val

        async def async_dummy_wrapper(_self: Any, *args: Any, **kwargs: Any) -> Any:
            return await f(_self, *args, **kwargs)

        def dummy_wrapper(_self: Any, *args: Any, **kwargs: Any) -> Any:
            return f(_self, *args, **kwargs)

        # check if already wrapped
        is_wrapped = getattr(f, "__wrapped__", False)
        if not is_wrapped:
            f.__wrapped__ = True  # type: ignore

        if asyncio.iscoroutinefunction(f):
            if is_wrapped:
                return async_dummy_wrapper
            else:
                return wrapped_async_llm_chat
        else:
            if is_wrapped:
                return dummy_wrapper
            else:
                return wrapped_llm_chat

    return wrap


def llm_completion_callback() -> Callable:
    def wrap(f: Callable) -> Callable:
        @contextmanager
        def wrapper_logic(_self: Any) -> Generator[CallbackManager, None, None]:
            callback_manager = getattr(_self, "callback_manager", None)
            if not isinstance(callback_manager, CallbackManager):
                raise ValueError(
                    "Cannot use llm_completion_callback on an instance "
                    "without a callback_manager attribute."
                )

            yield callback_manager

        async def wrapped_async_llm_predict(
            _self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            with wrapper_logic(_self) as callback_manager:
                event_id = callback_manager.on_event_start(
                    CBEventType.LLM,
                    payload={
                        EventPayload.PROMPT: args[0],
                        EventPayload.ADDITIONAL_KWARGS: kwargs,
                        EventPayload.SERIALIZED: _self.to_dict(),
                    },
                )

                f_return_val = await f(_self, *args, **kwargs)

                if isinstance(f_return_val, AsyncGenerator):
                    # intercept the generator and add a callback to the end
                    async def wrapped_gen() -> CompletionResponseAsyncGen:
                        last_response = None
                        async for x in f_return_val:
                            yield cast(CompletionResponse, x)
                            last_response = x

                        callback_manager.on_event_end(
                            CBEventType.LLM,
                            payload={
                                EventPayload.PROMPT: args[0],
                                EventPayload.COMPLETION: last_response,
                            },
                            event_id=event_id,
                        )

                    return wrapped_gen()
                else:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={
                            EventPayload.PROMPT: args[0],
                            EventPayload.RESPONSE: f_return_val,
                        },
                        event_id=event_id,
                    )

            return f_return_val

        def wrapped_llm_predict(_self: Any, *args: Any, **kwargs: Any) -> Any:
            with wrapper_logic(_self) as callback_manager:
                event_id = callback_manager.on_event_start(
                    CBEventType.LLM,
                    payload={
                        EventPayload.PROMPT: args[0],
                        EventPayload.ADDITIONAL_KWARGS: kwargs,
                        EventPayload.SERIALIZED: _self.to_dict(),
                    },
                )

                f_return_val = f(_self, *args, **kwargs)
                if isinstance(f_return_val, Generator):
                    # intercept the generator and add a callback to the end
                    def wrapped_gen() -> CompletionResponseGen:
                        last_response = None
                        for x in f_return_val:
                            yield cast(CompletionResponse, x)
                            last_response = x

                        callback_manager.on_event_end(
                            CBEventType.LLM,
                            payload={
                                EventPayload.PROMPT: args[0],
                                EventPayload.COMPLETION: last_response,
                            },
                            event_id=event_id,
                        )

                    return wrapped_gen()
                else:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={
                            EventPayload.PROMPT: args[0],
                            EventPayload.COMPLETION: f_return_val,
                        },
                        event_id=event_id,
                    )

            return f_return_val

        async def async_dummy_wrapper(_self: Any, *args: Any, **kwargs: Any) -> Any:
            return await f(_self, *args, **kwargs)

        def dummy_wrapper(_self: Any, *args: Any, **kwargs: Any) -> Any:
            return f(_self, *args, **kwargs)

        # check if already wrapped
        is_wrapped = getattr(f, "__wrapped__", False)
        if not is_wrapped:
            f.__wrapped__ = True  # type: ignore

        if asyncio.iscoroutinefunction(f):
            if is_wrapped:
                return async_dummy_wrapper
            else:
                return wrapped_async_llm_predict
        else:
            if is_wrapped:
                return dummy_wrapper
            else:
                return wrapped_llm_predict

    return wrap


class BaseLLM(ChainableMixin, BaseComponent):
    """LLM interface."""

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
        """LLM metadata."""

    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat endpoint for LLM."""

    @abstractmethod
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint for LLM."""

    @abstractmethod
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Streaming chat endpoint for LLM."""

    @abstractmethod
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming completion endpoint for LLM."""

    # ===== Async Endpoints =====
    @abstractmethod
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Async chat endpoint for LLM."""

    @abstractmethod
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Async completion endpoint for LLM."""

    @abstractmethod
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Async streaming chat endpoint for LLM."""

    @abstractmethod
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Async streaming completion endpoint for LLM."""
