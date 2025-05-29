import inspect
from typing import Callable, Generic, TypeVar, Union, Awaitable, cast, Dict, Any, Optional

T = TypeVar("T")

class _Resource(Generic[T]):
    def __init__(self, factory: Callable[..., Union[T, Awaitable[T]]], cache: bool) -> None:
        self._factory = factory
        self._is_async = inspect.iscoroutinefunction(factory)
        self.cache = cache
    async def call(self) -> T:
        if self._is_async:
            result = await cast(Callable[..., Awaitable[T]], self._factory)()
        else:
            result = cast(Callable[..., T], self._factory)()
        return result

def Resource(factory: Callable[..., T], cache: bool = True) -> _Resource[T]:
    return _Resource(factory, cache)

class ResourceManager:
    def __init__(self):
        self.resources: Dict[str, Any] = {}
    def put(self, name: str, resource: Any) -> None:
        self.resources.update({name: resource})
    def get(self, name: str, default: Optional[Any] = None) -> Any:
        if not self.resources.get(name, None):
            return default
        return self.resources.get(name)
