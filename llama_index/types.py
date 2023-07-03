from typing import Generator, Union, Protocol, Any, TypeVar
from abc import abstractmethod
from pydantic import BaseModel

Model = TypeVar("Model", bound=BaseModel)

TokenGen = Generator[str, None, None]
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
