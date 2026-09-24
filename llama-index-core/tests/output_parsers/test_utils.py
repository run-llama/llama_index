import pytest

from llama_index.core.output_parsers.base import OutputParserException
from llama_index.core.output_parsers.utils import (
    _marshal_llm_to_json,
    extract_json_str,
    parse_json_markdown,
)


def test_extract_json_str() -> None:
    input = """\
Here is the valid JSON:
{
    "title": "TestModel",
    "attr_dict": {
        "test_attr": "test_attr",
        "foo": 2
    }
}\
"""
    expected = """\
{
    "title": "TestModel",
    "attr_dict": {
        "test_attr": "test_attr",
        "foo": 2
    }
}\
"""
    assert extract_json_str(input) == expected


class TestMarshalLlmToJson:
    def test_valid_object(self) -> None:
        assert _marshal_llm_to_json('result: {"a": 1}') == '{"a": 1}'

    def test_valid_array(self) -> None:
        assert _marshal_llm_to_json("here: [1, 2, 3]") == "[1, 2, 3]"

    def test_no_brackets(self) -> None:
        assert _marshal_llm_to_json("just plain text") == ""

    def test_unclosed_object(self) -> None:
        text = 'Here is the result: {"choice": 1, "reason": "because'
        result = _marshal_llm_to_json(text)
        assert result.startswith("{")
        assert "choice" in result

    def test_unclosed_array(self) -> None:
        text = 'Here is the result: [{"choice": 1, "reason": "because'
        result = _marshal_llm_to_json(text)
        assert result.startswith("[")
        assert "choice" in result

    def test_closing_before_opening(self) -> None:
        text = "} some text { valid"
        result = _marshal_llm_to_json(text)
        assert result.startswith("{")


class TestParseJsonMarkdown:
    def test_valid_json(self) -> None:
        assert parse_json_markdown('{"a": 1}') == {"a": 1}

    def test_valid_json_with_markdown(self) -> None:
        assert parse_json_markdown('```json\n{"a": 1}\n```') == {"a": 1}

    def test_valid_array(self) -> None:
        assert parse_json_markdown("[1, 2]") == [1, 2]

    def test_trailing_comma_via_yaml(self) -> None:
        result = parse_json_markdown('{"a": 1,}')
        assert result == {"a": 1}

    def test_truncated_json_raises(self) -> None:
        text = 'Here is the result: [{"choice": 1, "reason": "because'
        with pytest.raises(OutputParserException):
            parse_json_markdown(text)

    def test_no_json_raises(self) -> None:
        with pytest.raises(OutputParserException, match="No JSON object"):
            parse_json_markdown("just plain text with no brackets")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(OutputParserException, match="No JSON object"):
            parse_json_markdown("")
