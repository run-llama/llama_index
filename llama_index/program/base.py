from typing import Any, Generic, TypeVar

from pydantic import BaseModel

Model = TypeVar("Model", bound=BaseModel)


class BasePydanticProgram(Generic[Model]):
    def __call__(self, *args: Any, **kwds: Any) -> Model:
        pass
