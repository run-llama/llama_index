"""ReAct output parser."""


from llama_index.types import BaseOutputParser
from typing import Any


class ReActOutputParser(BaseOutputParser):
    """ReAct Output parser."""

    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        return self._output_parser.parse(output)

    def format(self, output: str) -> str:
        """Format a query with structured output formatting instructions."""
        raise NotImplementedError
