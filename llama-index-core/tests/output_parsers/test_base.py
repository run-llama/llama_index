"""Test Output parsers."""

import pytest
from llama_index.core.output_parsers.langchain import LangchainOutputParser

try:
    import langchain  # pants: no-infer-dep
    from llama_index.core.bridge.langchain import (
        BaseOutputParser as LCOutputParser,
    )
    from llama_index.core.bridge.langchain import (
        ResponseSchema,
    )
except ImportError:
    langchain = None  # type: ignore


@pytest.mark.skipif(langchain is None, reason="langchain not installed")
def test_lc_output_parser() -> None:
    """Test langchain output parser."""

    class MockOutputParser(LCOutputParser):
        """
        Mock output parser.

        Similar to langchain's StructuredOutputParser, but better for testing.

        """

        response_schema: ResponseSchema

        def get_format_instructions(self) -> str:
            """Get format instructions."""
            return (
                f"{{ {self.response_schema.name}, {self.response_schema.description} }}"
            )

        def parse(self, text: str) -> str:
            """Parse the output of an LLM call."""
            # TODO: make this better
            return text

    response_schema = ResponseSchema(
        name="Education",
        description="education experience",
    )
    lc_output_parser = MockOutputParser(response_schema=response_schema)
    output_parser = LangchainOutputParser(lc_output_parser)

    query_str = "Hello world."
    output_instructions = output_parser.format(query_str)
    assert output_instructions == (
        "Hello world.\n\n{ Education, education experience }"
    )
    query_str = "foo {bar}."
    output_instructions = output_parser.format(query_str)
    assert output_instructions == (
        "foo {bar}.\n\n{{ Education, education experience }}"
    )
