from abc import ABC, abstractmethod
from typing import Any, Generic, Type, TypeVar

from pydantic import BaseModel

Model = TypeVar("Model", bound=BaseModel)


class BasePydanticProgram(ABC, Generic[Model]):
    @property
    @abstractmethod 
    def output_cls(self) -> Type[Model]:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Model:
        pass
