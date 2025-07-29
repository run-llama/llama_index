"""Base output parser class."""

from typing import TYPE_CHECKING, Any, Optional

from llama_index.core.output_parsers import BaseOutputParser
from llama_index.core.prompts.utils import SafeFormatter

if TYPE_CHECKING:
    from llama_index.core.bridge.langchain import (
        BaseOutputParser as LCOutputParser,
    )


class LangchainOutputParser(BaseOutputParser):
    """Langchain output parser."""

    def __init__(
        self, output_parser: "LCOutputParser", format_key: Optional[str] = None
    ) -> None:
        """Init params."""
        self._output_parser = output_parser
        self._format_key = format_key
        self._formatter = SafeFormatter()

    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        # Convert output to string if needed, then parse
        output_str = str(output) if not isinstance(output, str) else output
        return self._output_parser.parse(output_str)

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        format_instructions = self._output_parser.get_format_instructions()

        if self._format_key is not None:
            # Use SafeFormatter for query formatting
            self._formatter.format_dict = {self._format_key: format_instructions}
            fmt_query = self._formatter.format(query)
        else:
            fmt_query = query + "\n\n" + format_instructions

        return fmt_query
