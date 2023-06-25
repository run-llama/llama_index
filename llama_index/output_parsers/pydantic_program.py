"""Pydantic program output parser."""


from llama_index.output_parsers.base import BaseOutputParser
from llama_index.program.base_program import Model, BasePydanticProgram
from typing import Generic


DEFAULT_INPUT_PROMPT_TMPL = "{input_str}"


class PydanticProgramOutputParser(BaseOutputParser, Generic[Model]):
    """Pydantic Program output parser.

    Extracts text into a schema.

    """

    def __init__(
        self, pydantic_program: BasePydanticProgram[Model], input_key: str = "input_str"
    ) -> None:
        """Init params."""
        self._pydantic_program = pydantic_program
        self._input_key = input_key

    def parse(self, output: str) -> Model:
        """Parse, validate, and correct errors programmatically."""
        input_dict = {self._input_key: output}
        return self._pydantic_program(**input_dict)

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        raise NotImplementedError("`format` method not supported for Pydantic parser.")
