from typing import Annotated, Any, Generic, TypeVar, Union

from .events import StopEvent

StopEventT = TypeVar("StopEventT", bound=StopEvent)
# TODO: When releasing 1.0, remove support for Any
# and enforce usage of StopEventT
RunResultT = Union[StopEventT, Any]

T = TypeVar("T")


class _ResourceMeta(Generic[T]):
    def __class_getitem__(cls, item: tuple) -> Annotated[Any, Any]:
        if not(isinstance, tuple) or len(item) > 2:
            return TypeError("A resource should be defined as: Resource[type, factory]")
        type_t, factory = item
        if not callable(factory):
            raise TypeError(f"factory must be a callable, got {type(factory)}")
        return Annotated[type_t, factory]

Resource = _ResourceMeta
