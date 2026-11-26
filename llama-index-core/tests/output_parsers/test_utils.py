import json

import pytest

from llama_index.core.output_parsers.base import OutputParserException
from llama_index.core.output_parsers.utils import extract_json_str, parse_json_markdown


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


@pytest.mark.parametrize(
    "data",
    [
        {"text": "```json"},
        {"snippet": '```json\n{"value": 1}\n```'},
        {"nested": {"text": "before ```json after"}},
        ["```json", {"text": "```json"}],
        {"```json": "a marker in the key"},
    ],
)
def test_parse_json_markdown_preserves_fence_markers_in_payload(data: object) -> None:
    text = "```json\n" + json.dumps(data) + "\n```"

    assert parse_json_markdown(text) == data


@pytest.mark.parametrize(
    "text",
    [
        '{"value": 1}',
        '```json\n{"value": 1}\n```',
        'Here is the result:\n```json\n{"value": 1}\n```',
        '```json\r\n{"value": 1}\r\n```',
        '```json\n{"value": 1,}\n```',
    ],
)
def test_parse_json_markdown_existing_formats(text: str) -> None:
    assert parse_json_markdown(text) == {"value": 1}


@pytest.mark.parametrize("separator", ["\n", " "])
def test_parse_json_markdown_keeps_first_fenced_block(separator: str) -> None:
    first = '```json\n{"value": 1}\n```'
    second = '```json\n{"value": 2}\n```'

    assert parse_json_markdown(first + separator + second) == {"value": 1}


def test_parse_json_markdown_marker_payload_before_second_block() -> None:
    data = {"text": "```json"}
    first = "```json\n" + json.dumps(data) + "\n```"
    second = '```json\n{"value": 2}\n```'

    assert parse_json_markdown(first + "\n" + second) == data


@pytest.mark.parametrize(
    "suffix",
    [
        "Explanation: {not JSON}",
        '{"value": 2}',
        '```python\nvalue = {"value": 2}\n```',
    ],
)
def test_parse_json_markdown_preserves_existing_fallback(suffix: str) -> None:
    text = '```json\n{"value": 1}\n```\n' + suffix

    with pytest.raises(OutputParserException):
        parse_json_markdown(text)
