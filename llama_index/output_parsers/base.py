"""Base output parser class."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass
class StructuredOutput:
    """Structured output class."""

    raw_output: str
    parsed_output: Optional[Any] = None


class BaseOutputParser(Protocol):
    """Output parser class."""

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""

    @abstractmethod
    def format(self, output: str) -> str:
        """Format a query with structured output formatting instructions."""
