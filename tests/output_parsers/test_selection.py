import json.decoder

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


@pytest.mark.parametrize(
    ("output", "num_match"),
    [
        pytest.param(
            """[
    {"choice": 1, "reason": "just because"},
    {"choice": 2, "reason": "why not"}
]""",
            2,
            id="single_curly",
        ),
        pytest.param(
            """[
    {{"choice": 1, "reason": "just because"}},
    {{"choice": 2, "reason": "why not"}}
]""",
            2,
            id="double_curly",
        ),
        pytest.param(
            '\nOutput:\n[\n  {\n    "choice": 1,\n    "reason": "just because"\n  }\n]',
            1,
            id="https://github.com/jerryjliu/llama_index/issues/3135",
        ),
        pytest.param(
            """ Based on the given choices, the <shortened> question "<redacted>?" is:
(1) Useful for <redacted>
The reason for this choice is <redacted>. Therefore, option (1) is the most <shortened>
Here is the output in JSON format:
{{
  "type": "array",
  "items": {{
    "type": "object",
    "properties": {{
      "choice": 1,
      "reason": "just because"
    }},
    "required": [
      "choice",
      "reason"
    ],
    "additionalProperties": false
  }}
}}""",
            1,
            id="boss_fight",
        ),
    ],
)
def test_parse(
    output_parser: SelectionOutputParser, output: str, num_match: int
) -> None:
    parsed = output_parser.parse(output=output)
    assert isinstance(parsed, StructuredOutput)
    assert isinstance(parsed.parsed_output, list)
    assert len(parsed.parsed_output) == num_match
    assert parsed.parsed_output[0].choice == 1
    assert parsed.parsed_output[0].reason == "just because"


def test_failed_parse(output_parser: SelectionOutputParser) -> None:
    no_json_in_response = (
        " Based on the given choices, the most relevant choice for the question"
        " 'What are the <redacted>?' is:\n\n(1) <redacted>.\n\nThe reason for"
        " this choice is that <redacted>. Therefore, choosing option (1) would"
        " provide the most relevant information for finding the <redacted>."
    )
    with pytest.raises(ValueError, match="Failed to convert") as exc_info:
        output_parser.parse(output=no_json_in_response)
    assert isinstance(exc_info.value.__cause__, json.decoder.JSONDecodeError)
