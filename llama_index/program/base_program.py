from abc import ABC, abstractmethod
from typing import Any, Generic, Type, TypeVar
from llama_index.types import Model

from pydantic import BaseModel


class BasePydanticProgram(ABC, Generic[Model]):
    """A base class for LLM-powered function that return a pydantic model.

    Note: this interface is not yet stable.
    """

    @classmethod
    @abstractmethod
    def from_defaults(
        cls, output_cls: Type[Model], prompt_template_str: str, **kwargs: Any
    ) -> "BasePydanticProgram":
        """Create a default pydantic program."""
        raise NotImplementedError

    @property
    @abstractmethod
    def output_cls(self) -> Type[Model]:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Model:
        pass
