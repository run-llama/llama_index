from llama_index.core.llms.utils import parse_partial_json


def test_parse_partial_json_incomplete_key():
    s = '{"key'
    result = parse_partial_json(s)
    assert result == {}


def test_parse_partial_json_incomplete_string_value():
    s = '{"key": "value'
    result = parse_partial_json(s)
    assert result["key"] == "value"
