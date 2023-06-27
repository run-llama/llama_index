from typing import Generator, Union
from typing import Protocol, Any
from abc import abstractmethod


StreamTokens = Generator[str, None, None]
RESPONSE_TEXT_TYPE = Union[str, StreamTokens]


# TODO: move into a `core` folder
class BaseOutputParser(Protocol):
    """Output parser class."""

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""

    @abstractmethod
    def format(self, output: str) -> str:
        """Format a query with structured output formatting instructions."""
