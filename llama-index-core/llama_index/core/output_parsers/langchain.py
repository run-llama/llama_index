"""Base output parser class."""

from string import Formatter
from typing import TYPE_CHECKING, Any, Optional

from llama_index.core.output_parsers.base import ChainableOutputParser

if TYPE_CHECKING:
    from llama_index.core.bridge.langchain import (
        BaseOutputParser as LCOutputParser,
    )


class LangchainOutputParser(ChainableOutputParser):
    """Langchain output parser."""

    def __init__(
        self, output_parser: "LCOutputParser", format_key: Optional[str] = None
    ) -> None:
        """Init params."""
        self._output_parser = output_parser
        self._format_key = format_key

    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        # Convert output to string if it isn't already, handling various input types
        if not isinstance(output, str):
            try:
                output = str(output).strip()
            except Exception as e:
                raise ValueError(f"Unable to convert output to string: {e}")
        
        return self._output_parser.parse(output)

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        format_instructions = self._output_parser.get_format_instructions()

        query_tmpl_vars = {
            v for _, v, _, _ in Formatter().parse(query) if v is not None
        }
        
        if len(query_tmpl_vars) > 0:
            format_instructions = format_instructions.replace("{{", "{{{{")
            format_instructions = format_instructions.replace("}}", "}}}}")
            format_instructions = format_instructions.replace("{", "{{")
            format_instructions = format_instructions.replace("}", "}}")

        if self._format_key is not None:
            fmt_query = query.format(**{self._format_key: format_instructions})
        else:
            fmt_query = query + "\n\n" + format_instructions

        return fmt_query
