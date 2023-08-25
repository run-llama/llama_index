"""Base output parser class."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class StructuredOutput:
    """Structured output class."""

    raw_output: str
    parsed_output: Optional[Any] = None


class OutputParserException(Exception):
    pass
