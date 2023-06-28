"""Pydantic output parser."""

from llama_index.types import BaseOutputParser, Model
from typing import Type, Optional, List, Any
import re
import json


class PydanticOutputParser(BaseOutputParser):
    """Pydantic Output Parser.

    Args:
        output_cls (BaseModel): Pydantic output class.

    """

    def __init__(
        self,
        output_cls: Type[Model],
        excluded_schema_keys_from_format: Optional[List] = None,
    ) -> None:
        """Init params."""
        self._output_cls = output_cls
        self._excluded_schema_keys_from_format = excluded_schema_keys_from_format or []

    @property
    def output_cls(self) -> Type[Model]:
        return self._output_cls

    def parse(self, text: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
        match = re.search(
            r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        json_str = match.group() if match else ""
        return self._output_cls.parse_raw(json_str)

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        schema_dict = self._output_cls.schema()
        for key in self._excluded_schema_keys_from_format:
            del schema_dict[key]

        return query + "\n\n" + json.dumps(schema_dict, indent=2)
