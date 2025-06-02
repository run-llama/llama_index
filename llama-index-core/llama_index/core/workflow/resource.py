import inspect
from typing import (
    Callable,
    Generic,
    TypeVar,
    Union,
    Awaitable,
    Dict,
    Any,
    cast,
)
from pydantic import (
    BaseModel,
    ConfigDict,
)

T = TypeVar("T")


class _Resource(Generic[T]):
    def __init__(
        self, factory: Callable[..., Union[T, Awaitable[T]]], cache: bool
    ) -> None:
        self._factory = factory
        self._is_async = inspect.iscoroutinefunction(factory)
        self.name = factory.__qualname__
        self.cache = cache

    async def call(self) -> T:
        if self._is_async:
            result = await cast(Callable[..., Awaitable[T]], self._factory)()
        else:
            result = cast(Callable[..., T], self._factory)()
        return result


class ResourceDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    resource: _Resource


def Resource(factory: Callable[..., T], cache: bool = True) -> _Resource[T]:
    return _Resource(factory, cache)


class ResourceManager:
    def __init__(self) -> None:
        self.resources: Dict[str, Any] = {}

    async def set(self, name: str, val: Any) -> None:
        self.resources.update({name: val})

    async def get(self, resource: _Resource) -> Any:
        if not resource.cache:
            val = await resource.call()
        elif resource.cache and not self.resources.get(resource.name, None):
            val = await resource.call()
            await self.set(resource.name, val)
        else:
            val = self.resources.get(resource.name)
        return val

    def get_all(self) -> Dict[str, Any]:
        return self.resources
