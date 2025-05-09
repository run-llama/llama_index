import asyncio
import functools
import logging
from typing import Any, Callable, cast

from llama_index.core.callbacks.base import CallbackManager

logger = logging.getLogger(__name__)


def trace_method(
    trace_id: str, callback_manager_attr: str = "callback_manager"
) -> Callable[[Callable], Callable]:
    """
    Decorator to trace a method.

    Example:
        @trace_method("my_trace_id")
        def my_method(self):
            pass

    Assumes that the self instance has a CallbackManager instance in an attribute
    named `callback_manager`.
    This can be overridden by passing in a `callback_manager_attr` keyword argument.

    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)  # preserve signature, name, etc. of func
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                callback_manager = getattr(self, callback_manager_attr)
            except AttributeError:
                logger.warning(
                    "Could not find attribute %s on %s.",
                    callback_manager_attr,
                    type(self),
                )
                return func(self, *args, **kwargs)
            callback_manager = cast(CallbackManager, callback_manager)
            with callback_manager.as_trace(trace_id):
                return func(self, *args, **kwargs)

        @functools.wraps(func)  # preserve signature, name, etc. of func
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                callback_manager = getattr(self, callback_manager_attr)
            except AttributeError:
                logger.warning(
                    "Could not find attribute %s on %s.",
                    callback_manager_attr,
                    type(self),
                )
                return await func(self, *args, **kwargs)
            callback_manager = cast(CallbackManager, callback_manager)
            with callback_manager.as_trace(trace_id):
                return await func(self, *args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator
