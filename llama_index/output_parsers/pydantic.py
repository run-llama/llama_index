"""Pydantic output parser."""

import json
from typing import Any, List, Optional, Tuple, Type

from llama_index.output_parsers.base import ChainableOutputParser
from llama_index.output_parsers.utils import extract_json_str
from llama_index.types import Model

PYDANTIC_FORMAT_TMPL = """
Here's a JSON schema to follow:
{schema}

Output a valid JSON object but do not repeat the schema.
{examples}
Input: {query}\n
Output:\n
"""


class PydanticOutputParser(ChainableOutputParser):
    """Pydantic Output Parser.

    Args:
        output_cls (BaseModel): Pydantic output class.

    """

    def __init__(
        self,
        output_cls: Type[Model],
        excluded_schema_keys_from_format: Optional[List] = None,
        pydantic_format_tmpl: str = PYDANTIC_FORMAT_TMPL,
        examples: Optional[List[Tuple[str, Type[Model]]]] = None,
    ) -> None:
        """Init params."""
        self._output_cls = output_cls
        self._excluded_schema_keys_from_format = excluded_schema_keys_from_format or []
        self._pydantic_format_tmpl = pydantic_format_tmpl
        self._examples = examples or []

    @property
    def output_cls(self) -> Type[Model]:
        return self._output_cls

    @property
    def format_string(self) -> str:
        """Format string."""
        return self.get_format_string(escape_json=True, query="{query}")

    def get_format_string(self, query: str, escape_json: bool = True) -> str:
        """Format string."""
        schema_dict = self._output_cls.schema()
        for key in self._excluded_schema_keys_from_format:
            del schema_dict[key]

        schema_str = json.dumps(schema_dict)

        if len(self._examples) > 0:
            examples_str = "\n\n".join(
                [
                    f"Input Example {i + 1}:\n{input_str}\n\nOutput Example {i+1}:\n{json.dumps(example_cls.dict())}\n\n"
                    for i, (input_str, example_cls) in enumerate(self._examples)
                ]
            )
        else:
            examples_str = ""

        output_str = self._pydantic_format_tmpl.format(
            schema=schema_str, examples=examples_str, query=query
        )

        if escape_json:
            return output_str.replace("{", "{{").replace("}", "}}")
        else:
            return output_str

    def parse(self, text: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        json_str = extract_json_str(text)
        return self._output_cls.parse_raw(json_str)

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        return self.get_format_string(query, escape_json=False)
