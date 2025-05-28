from typing import Any, Callable, Generic, TypeVar, Union

from .events import StopEvent

StopEventT = TypeVar("StopEventT", bound=StopEvent)
# TODO: When releasing 1.0, remove support for Any
# and enforce usage of StopEventT
RunResultT = Union[StopEventT, Any]

T = TypeVar("T")


class _Resource(Generic[T]):
    def __init__(self, factory: Callable[..., T], cache: bool) -> None:
        self._factory = factory
        self._cache = cache
        self._cached_value: T | None = None

    def __call__(self) -> T:
        if self._cache:
            if self._cached_value is None:
                self._cached_value = self._factory()
            return self._cached_value
        return self._factory()


def Resource(factory: Callable[..., T]) -> _Resource[T]:
    return _Resource(factory, True)
