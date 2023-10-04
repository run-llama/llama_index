"""Pydantic output parser."""

import json
from typing import Any, List, Optional, Type

from llama_index.output_parsers.utils import extract_json_str
from llama_index.types import BaseOutputParser, Model

PYDANTIC_FORMAT_TMPL = """
Here's a JSON schema to follow:
{schema}

Output a valid JSON object but do not repeat the schema.
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

    @property
    def format_string(self) -> str:
        schema_dict = self._output_cls.schema()
        for key in self._excluded_schema_keys_from_format:
            del schema_dict[key]

        schema_str = json.dumps(schema_dict)
        return self._pydantic_format_tmpl.format(schema=schema_str)

    def parse(self, text: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        json_str = extract_json_str(text)
        return self._output_cls.parse_raw(json_str)

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        return query + "\n\n" + self.format_string
