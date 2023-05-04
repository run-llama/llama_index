import pytest
from llama_index.output_parsers.base import StructuredOutput
from llama_index.output_parsers.selection import SelectionOutputParser


@pytest.fixture()
def output_parser() -> SelectionOutputParser:
    return SelectionOutputParser()


def test_format(output_parser: SelectionOutputParser) -> None:
    test_template = "Test prompt template with some {field} to fill in."
    new_test_template = output_parser.format(test_template)
    new_test_template.format(field="field")


def test_parse(output_parser: SelectionOutputParser) -> None:
    output = """[
        {"choice": 1, "reason": "just because"},
        {"choice": 2, "reason": "why not"}
    ]
    """
    parsed = output_parser.parse(output=output)
    assert isinstance(parsed, StructuredOutput)
    assert isinstance(parsed.parsed_output, list)
    assert len(parsed.parsed_output) == 2
    assert parsed.parsed_output[0].choice == 1
    assert parsed.parsed_output[0].reason == "just because"
