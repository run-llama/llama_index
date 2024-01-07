"""Base output parser class."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from llama_index.bridge.pydantic import Field
from llama_index.core.query_pipeline.query_component import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from llama_index.types import BaseOutputParser


@dataclass
class StructuredOutput:
    """Structured output class."""

    raw_output: str
    parsed_output: Optional[Any] = None


class OutputParserException(Exception):
    pass


class ChainableOutputParser(BaseOutputParser, ChainableMixin):
    """Chainable output parser."""

    # TODO: consolidate with base at some point if possible.

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """Get query component."""
        return OutputParserComponent(output_parser=self)


class OutputParserComponent(QueryComponent):
    """Output parser component."""

    output_parser: BaseOutputParser = Field(..., description="Output parser.")

    class Config:
        arbitrary_types_allowed = True

    def _run_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run component."""
        output = self.output_parser.parse(kwargs["input"])
        return {"output": output}

    async def _arun_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run component."""
        # NOTE: no native async for output parser
        return self._run_component(**kwargs)

    def _validate_component_inputs(self, input: Any) -> Any:
        """Validate component inputs during run_component."""
        input["input"] = validate_and_convert_stringable(input["input"])
        return input

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""

    @property
    def input_keys(self) -> Any:
        """Input keys."""
        return InputKeys.from_keys({"input"})

    @property
    def output_keys(self) -> Any:
        """Output keys."""
        return OutputKeys.from_keys({"output"})
