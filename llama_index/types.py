from abc import abstractmethod
from typing import Any, AsyncGenerator, Generator, Protocol, TypeVar, Union

from pydantic.v1 import BaseModel

Model = TypeVar("Model", bound=BaseModel)

TokenGen = Generator[str, None, None]
TokenAsyncGen = AsyncGenerator[str, None]
RESPONSE_TEXT_TYPE = Union[str, TokenGen]


# TODO: move into a `core` folder
class BaseOutputParser(Protocol):
    """Output parser class."""

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""

    @abstractmethod
    def format(self, output: str) -> str:
        """Format a query with structured output formatting instructions."""
