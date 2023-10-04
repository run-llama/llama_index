"""Base output parser class."""

from string import Formatter
from typing import Any, Optional

from llama_index.bridge.langchain import BaseOutputParser as LCOutputParser
from llama_index.types import BaseOutputParser


class LangchainOutputParser(BaseOutputParser):
    """Langchain output parser."""

    def __init__(
        self, output_parser: LCOutputParser, format_key: Optional[str] = None
    ) -> None:
        """Init params."""
        self._output_parser = output_parser
        self._format_key = format_key

    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        # TODO: this object may be stringified by our upstream llmpredictor,
        # figure out better
        # ways to "convert" the object to a proper string format.
        return self._output_parser.parse(output)

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        format_instructions = self._output_parser.get_format_instructions()

        # TODO: this is a temporary hack. if there's curly brackets in the format
        # instructions (and query is a string template), we need to
        # escape the curly brackets in the format instructions to preserve the
        # overall template.
        query_tmpl_vars = {
            v for _, v, _, _ in Formatter().parse(query) if v is not None
        }
        if len(query_tmpl_vars) > 0:
            format_instructions = format_instructions.replace("{", "{{")
            format_instructions = format_instructions.replace("}", "}}")

        if self._format_key is not None:
            fmt_query = query.format(**{self._format_key: format_instructions})
        else:
            fmt_query = query + "\n\n" + format_instructions

        return fmt_query
