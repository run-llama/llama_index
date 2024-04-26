import pytest

from llama_index.program.openai.utils import parse_partial_json


def test_valid_partial_json():
    assert parse_partial_json("{") == {}
    with pytest.raises(ValueError):
        parse_partial_json('{"foo":')
    assert parse_partial_json('{"foo": "bar') == {"foo": "bar"}


def test_invalid_partial_json():
    with pytest.raises(ValueError):
        parse_partial_json("foo")


def test_partial_json_with_string():
    assert parse_partial_json('{"foo": "') == {"foo": ""}


def test_nested_brackets():
    assert parse_partial_json('{"foo": [1, 2, 3') == {"foo": [1, 2, 3]}
    with pytest.raises(ValueError):
        parse_partial_json('{"foo": [1, 2, 3}')


def test_escape_sequences():
    assert parse_partial_json('{"foo": "new\\nline"}') == {"foo": "new\nline"}


def test_nested_objects():
    assert parse_partial_json('{"foo": {"bar":{') == {"foo": {"bar": {}}}
    with pytest.raises(ValueError):
        parse_partial_json('{"foo": {"bar":')


def test_mismatched_closing_brackets():
    with pytest.raises(ValueError):
        parse_partial_json('{"foo": []}}')


def test_complex_partial_json():
    assert parse_partial_json(
        '{"foo": [1, {"bar": "baz"}, {"qux": ["hello", "world"]}]}'
    ) == {"foo": [1, {"bar": "baz"}, {"qux": ["hello", "world"]}]}
