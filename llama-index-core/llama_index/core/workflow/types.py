import inspect
from typing import Any, Callable, Generic, TypeVar, Union, Awaitable, cast

from .events import StopEvent

StopEventT = TypeVar("StopEventT", bound=StopEvent)
# TODO: When releasing 1.0, remove support for Any
# and enforce usage of StopEventT
RunResultT = Union[StopEventT, Any]

T = TypeVar("T")


class _Resource(Generic[T]):
    def __init__(self, factory: Callable[..., Union[T, Awaitable[T]]], cache: bool) -> None:
        self._factory = factory
        self._cache = cache
        self._cached_value: T | None = None
        self._is_async = inspect.iscoroutinefunction(factory)

    async def call(self) -> T:
        if self._cached_value is not None:
            return self._cached_value
        if self._is_async:
            result = await cast(Callable[..., Awaitable[T]], self._factory)()
        else:
            result = cast(Callable[..., T], self._factory)()
        if self._cache:
            self._cached_value = result
        return result

def Resource(factory: Callable[..., T], cache: bool = True) -> _Resource[T]:
    return _Resource(factory, cache)
