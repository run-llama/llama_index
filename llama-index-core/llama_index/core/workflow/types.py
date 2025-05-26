from typing import Annotated, Any, Generic, TypeVar, Union

from .events import StopEvent

StopEventT = TypeVar("StopEventT", bound=StopEvent)
# TODO: When releasing 1.0, remove support for Any
# and enforce usage of StopEventT
RunResultT = Union[StopEventT, Any]

T = TypeVar("T")


class _ResourceMeta(Generic[T]):
    def __class_getitem__(cls, item):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Resource must be used as Resource[type, factory_function]")
        t_type, factory = item
        return Annotated[t_type, factory]


Resource = _ResourceMeta
