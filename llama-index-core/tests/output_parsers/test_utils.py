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


def test_marshal_unclosed_array_keeps_truncated_payload() -> None:
    output = 'Here is the result: [{"choice": 1, "reason": "because'
    assert _marshal_llm_to_json(output) == '[{"choice": 1, "reason": "because'


def test_marshal_unclosed_object_keeps_truncated_payload() -> None:
    output = 'Here is the result: {"a": 1, "b":'
    assert _marshal_llm_to_json(output) == '{"a": 1, "b":'


def test_parse_json_markdown_truncated_array_raises() -> None:
    truncated = 'Here is the result: [{"choice": 1, "reason": "because'
    with pytest.raises(OutputParserException) as exc_info:
        parse_json_markdown(truncated)
    assert '[{"choice": 1, "reason": "because' in str(exc_info.value)


def test_parse_json_markdown_truncated_object_raises() -> None:
    with pytest.raises(OutputParserException):
        parse_json_markdown('Here is the result: {"a": 1, "b":')


def test_parse_json_markdown_without_json_raises() -> None:
    with pytest.raises(OutputParserException):
        parse_json_markdown("there is no json in this answer")


def test_parse_json_markdown_closed_array() -> None:
    output = 'Here is the result: [{"choice": 1, "reason": "because"}]'
    assert parse_json_markdown(output) == [{"choice": 1, "reason": "because"}]
