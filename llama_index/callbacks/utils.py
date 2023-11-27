import asyncio
import dataclasses
import json
import logging
from datetime import date, datetime
from typing import Any, Callable, Generator, cast

from pydantic import BaseModel

from llama_index.callbacks.base import CallbackManager

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


class SuperJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return self.default(dataclasses.asdict(o))
        if isinstance(o, BaseModel):
            return self.default(o.model_dump())
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, Exception):
            return str(o)
        if isinstance(o, str):
            return o
        if isinstance(o, list):
            return [self.default(x) for x in o]
        if isinstance(o, dict):
            return {k: self.default(v) for k, v in o.items()}
        if isinstance(o, Generator):
            return None
        return super().default(o)


def superjson_dumps(data):
    return json.dumps(data, cls=SuperJSONEncoder)
