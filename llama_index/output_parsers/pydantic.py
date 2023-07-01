"""Pydantic output parser."""

from llama_index.types import BaseOutputParser, Model
from typing import Type, Optional, List, Any
import re
import json

PYDANTIC_FORMAT_TMPL = """
Please use the following JSON schema to format your query:
{schema}
"""


class PydanticOutputParser(BaseOutputParser):
    """Pydantic Output Parser.

    Args:
        output_cls (BaseModel): Pydantic output class.

    """

    def __init__(
        self,
        output_cls: Type[Model],
        excluded_schema_keys_from_format: Optional[List] = None,
        pydantic_format_tmpl: str = PYDANTIC_FORMAT_TMPL,
    ) -> None:
        """Init params."""
        self._output_cls = output_cls
        self._excluded_schema_keys_from_format = excluded_schema_keys_from_format or []
        self._pydantic_format_tmpl = pydantic_format_tmpl

    @property
    def output_cls(self) -> Type[Model]:
        return self._output_cls

    def parse(self, text: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
        match = re.search(
            r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        if match:
            json_str = match.group()
            return self._output_cls.parse_raw(json_str)
        else:
            raise ValueError(f"Could not parse output: {text}")

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        schema_dict = self._output_cls.schema()
        for key in self._excluded_schema_keys_from_format:
            del schema_dict[key]

        schema_str = json.dumps(schema_dict)
        # escape left and right brackets with double brackets
        schema_str = schema_str.replace("{", "{{")
        schema_str = schema_str.replace("}", "}}")
        format_str = self._pydantic_format_tmpl.format(schema=schema_str)

        return query + "\n\n" + format_str
