import asyncio
from contextlib import contextmanager
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Sequence,
    cast,
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload

# dispatcher setup
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.exception import ExceptionEvent
from llama_index.core.instrumentation.span import active_span_id
from llama_index.core.instrumentation.events.llm import (
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMChatInProgressEvent,
    LLMCompletionInProgressEvent,
)

dispatcher = get_dispatcher(__name__)


def llm_chat_callback() -> Callable:
    def wrap(f: Callable) -> Callable:
        @contextmanager
        def wrapper_logic(_self: Any) -> Generator[CallbackManager, None, None]:
            callback_manager = getattr(_self, "callback_manager", None)
            if not isinstance(callback_manager, CallbackManager):
                _self.callback_manager = CallbackManager()

            yield _self.callback_manager  # type: ignore

        async def wrapped_async_llm_chat(
            _self: Any, messages: Sequence[ChatMessage], **kwargs: Any
        ) -> Any:
            with wrapper_logic(_self) as callback_manager, callback_manager.as_trace(
                "chat"
            ):
                span_id = active_span_id.get()
                model_dict = _self.to_dict()
                model_dict.pop("api_key", None)
                dispatcher.event(
                    LLMChatStartEvent(
                        model_dict=model_dict,
                        messages=messages,
                        additional_kwargs=kwargs,
                        span_id=span_id,
                    )
                )
                event_id = callback_manager.on_event_start(
                    CBEventType.LLM,
                    payload={
                        EventPayload.MESSAGES: messages,
                        EventPayload.ADDITIONAL_KWARGS: kwargs,
                        EventPayload.SERIALIZED: _self.to_dict(),
                    },
                )
                try:
                    f_return_val = await f(_self, messages, **kwargs)
                except BaseException as e:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={EventPayload.EXCEPTION: e},
                        event_id=event_id,
                    )
                    raise
                if isinstance(f_return_val, AsyncGenerator):
                    # intercept the generator and add a callback to the end
                    async def wrapped_gen() -> ChatResponseAsyncGen:
                        last_response = None
                        try:
                            async for x in f_return_val:
                                dispatcher.event(
                                    LLMChatInProgressEvent(
                                        messages=messages,
                                        response=x,
                                        span_id=span_id,
                                    )
                                )
                                yield cast(ChatResponse, x)
                                last_response = x
                        except BaseException as exception:
                            callback_manager.on_event_end(
                                CBEventType.LLM,
                                payload={EventPayload.EXCEPTION: exception},
                                event_id=event_id,
                            )
                            dispatcher.event(
                                ExceptionEvent(
                                    exception=exception,
                                    span_id=span_id,
                                )
                            )
                            raise
                        callback_manager.on_event_end(
                            CBEventType.LLM,
                            payload={
                                EventPayload.MESSAGES: messages,
                                EventPayload.RESPONSE: last_response,
                            },
                            event_id=event_id,
                        )
                        dispatcher.event(
                            LLMChatEndEvent(
                                messages=messages,
                                response=last_response,
                                span_id=span_id,
                            )
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
                    dispatcher.event(
                        LLMChatEndEvent(
                            messages=messages,
                            response=f_return_val,
                            span_id=span_id,
                        )
                    )

            return f_return_val

        def wrapped_llm_chat(
            _self: Any, messages: Sequence[ChatMessage], **kwargs: Any
        ) -> Any:
            with wrapper_logic(_self) as callback_manager, callback_manager.as_trace(
                "chat"
            ):
                span_id = active_span_id.get()
                model_dict = _self.to_dict()
                model_dict.pop("api_key", None)
                dispatcher.event(
                    LLMChatStartEvent(
                        model_dict=model_dict,
                        messages=messages,
                        additional_kwargs=kwargs,
                        span_id=span_id,
                    )
                )
                event_id = callback_manager.on_event_start(
                    CBEventType.LLM,
                    payload={
                        EventPayload.MESSAGES: messages,
                        EventPayload.ADDITIONAL_KWARGS: kwargs,
                        EventPayload.SERIALIZED: _self.to_dict(),
                    },
                )
                try:
                    f_return_val = f(_self, messages, **kwargs)
                except BaseException as e:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={EventPayload.EXCEPTION: e},
                        event_id=event_id,
                    )
                    raise
                if isinstance(f_return_val, Generator):
                    # intercept the generator and add a callback to the end
                    def wrapped_gen() -> ChatResponseGen:
                        last_response = None
                        try:
                            for x in f_return_val:
                                dispatcher.event(
                                    LLMChatInProgressEvent(
                                        messages=messages,
                                        response=x,
                                        span_id=span_id,
                                    )
                                )
                                yield cast(ChatResponse, x)
                                last_response = x
                        except BaseException as exception:
                            callback_manager.on_event_end(
                                CBEventType.LLM,
                                payload={EventPayload.EXCEPTION: exception},
                                event_id=event_id,
                            )
                            dispatcher.event(
                                ExceptionEvent(
                                    exception=exception,
                                    span_id=span_id,
                                )
                            )
                            raise
                        callback_manager.on_event_end(
                            CBEventType.LLM,
                            payload={
                                EventPayload.MESSAGES: messages,
                                EventPayload.RESPONSE: last_response,
                            },
                            event_id=event_id,
                        )
                        dispatcher.event(
                            LLMChatEndEvent(
                                messages=messages,
                                response=last_response,
                                span_id=span_id,
                            )
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
                    dispatcher.event(
                        LLMChatEndEvent(
                            messages=messages,
                            response=f_return_val,
                            span_id=span_id,
                        )
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

        # Update the wrapper function to look like the wrapped function.
        # See e.g. https://github.com/python/cpython/blob/0abf997e75bd3a8b76d920d33cc64d5e6c2d380f/Lib/functools.py#L57
        for attr in (
            "__module__",
            "__name__",
            "__qualname__",
            "__doc__",
            "__annotations__",
            "__type_params__",
        ):
            if v := getattr(f, attr, None):
                setattr(async_dummy_wrapper, attr, v)
                setattr(wrapped_async_llm_chat, attr, v)
                setattr(dummy_wrapper, attr, v)
                setattr(wrapped_llm_chat, attr, v)

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
                _self.callback_manager = CallbackManager()

            yield _self.callback_manager

        def extract_prompt(*args: Any, **kwargs: Any) -> str:
            if len(args) > 0:
                return str(args[0])
            elif "prompt" in kwargs:
                return kwargs["prompt"]
            else:
                raise ValueError(
                    "No prompt provided in positional or keyword arguments"
                )

        async def wrapped_async_llm_predict(
            _self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            prompt = extract_prompt(*args, **kwargs)
            with wrapper_logic(_self) as callback_manager, callback_manager.as_trace(
                "completion"
            ):
                span_id = active_span_id.get()
                model_dict = _self.to_dict()
                model_dict.pop("api_key", None)
                dispatcher.event(
                    LLMCompletionStartEvent(
                        model_dict=model_dict,
                        prompt=prompt,
                        additional_kwargs=kwargs,
                        span_id=span_id,
                    )
                )
                event_id = callback_manager.on_event_start(
                    CBEventType.LLM,
                    payload={
                        EventPayload.PROMPT: prompt,
                        EventPayload.ADDITIONAL_KWARGS: kwargs,
                        EventPayload.SERIALIZED: _self.to_dict(),
                    },
                )

                try:
                    f_return_val = await f(_self, *args, **kwargs)
                except BaseException as e:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={EventPayload.EXCEPTION: e},
                        event_id=event_id,
                    )
                    raise
                if isinstance(f_return_val, AsyncGenerator):
                    # intercept the generator and add a callback to the end
                    async def wrapped_gen() -> CompletionResponseAsyncGen:
                        last_response = None
                        try:
                            async for x in f_return_val:
                                dispatcher.event(
                                    LLMCompletionInProgressEvent(
                                        prompt=prompt,
                                        response=x,
                                        span_id=span_id,
                                    )
                                )
                                yield cast(CompletionResponse, x)
                                last_response = x
                        except BaseException as exception:
                            callback_manager.on_event_end(
                                CBEventType.LLM,
                                payload={EventPayload.EXCEPTION: exception},
                                event_id=event_id,
                            )
                            dispatcher.event(
                                ExceptionEvent(
                                    exception=exception,
                                    span_id=span_id,
                                )
                            )
                            raise
                        callback_manager.on_event_end(
                            CBEventType.LLM,
                            payload={
                                EventPayload.PROMPT: prompt,
                                EventPayload.COMPLETION: last_response,
                            },
                            event_id=event_id,
                        )
                        dispatcher.event(
                            LLMCompletionEndEvent(
                                prompt=prompt,
                                response=last_response,
                                span_id=span_id,
                            )
                        )

                    return wrapped_gen()
                else:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={
                            EventPayload.PROMPT: prompt,
                            EventPayload.COMPLETION: f_return_val,
                        },
                        event_id=event_id,
                    )
                    dispatcher.event(
                        LLMCompletionEndEvent(
                            prompt=prompt,
                            response=f_return_val,
                            span_id=span_id,
                        )
                    )

            return f_return_val

        def wrapped_llm_predict(_self: Any, *args: Any, **kwargs: Any) -> Any:
            prompt = extract_prompt(*args, **kwargs)
            with wrapper_logic(_self) as callback_manager, callback_manager.as_trace(
                "completion"
            ):
                span_id = active_span_id.get()
                model_dict = _self.to_dict()
                model_dict.pop("api_key", None)
                dispatcher.event(
                    LLMCompletionStartEvent(
                        model_dict=model_dict,
                        prompt=prompt,
                        additional_kwargs=kwargs,
                        span_id=span_id,
                    )
                )
                event_id = callback_manager.on_event_start(
                    CBEventType.LLM,
                    payload={
                        EventPayload.PROMPT: prompt,
                        EventPayload.ADDITIONAL_KWARGS: kwargs,
                        EventPayload.SERIALIZED: _self.to_dict(),
                    },
                )
                try:
                    f_return_val = f(_self, *args, **kwargs)
                except BaseException as e:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={EventPayload.EXCEPTION: e},
                        event_id=event_id,
                    )
                    raise
                if isinstance(f_return_val, Generator):
                    # intercept the generator and add a callback to the end
                    def wrapped_gen() -> CompletionResponseGen:
                        last_response = None
                        try:
                            for x in f_return_val:
                                dispatcher.event(
                                    LLMCompletionInProgressEvent(
                                        prompt=prompt,
                                        response=x,
                                        span_id=span_id,
                                    )
                                )
                                yield cast(CompletionResponse, x)
                                last_response = x
                        except BaseException as exception:
                            callback_manager.on_event_end(
                                CBEventType.LLM,
                                payload={EventPayload.EXCEPTION: exception},
                                event_id=event_id,
                            )
                            dispatcher.event(
                                ExceptionEvent(
                                    exception=exception,
                                    span_id=span_id,
                                )
                            )
                            raise
                        callback_manager.on_event_end(
                            CBEventType.LLM,
                            payload={
                                EventPayload.PROMPT: prompt,
                                EventPayload.COMPLETION: last_response,
                            },
                            event_id=event_id,
                        )
                        dispatcher.event(
                            LLMCompletionEndEvent(
                                prompt=prompt,
                                response=last_response,
                                span_id=span_id,
                            )
                        )

                    return wrapped_gen()
                else:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={
                            EventPayload.PROMPT: prompt,
                            EventPayload.COMPLETION: f_return_val,
                        },
                        event_id=event_id,
                    )
                    dispatcher.event(
                        LLMCompletionEndEvent(
                            prompt=prompt,
                            response=f_return_val,
                            span_id=span_id,
                        )
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

        # Update the wrapper function to look like the wrapped function.
        # See e.g. https://github.com/python/cpython/blob/0abf997e75bd3a8b76d920d33cc64d5e6c2d380f/Lib/functools.py#L57
        for attr in (
            "__module__",
            "__name__",
            "__qualname__",
            "__doc__",
            "__annotations__",
            "__type_params__",
        ):
            if v := getattr(f, attr, None):
                setattr(async_dummy_wrapper, attr, v)
                setattr(wrapped_async_llm_predict, attr, v)
                setattr(dummy_wrapper, attr, v)
                setattr(wrapped_llm_predict, attr, v)

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
