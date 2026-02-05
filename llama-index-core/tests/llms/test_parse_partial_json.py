import pytest
from llama_index.core.llms.utils import parse_partial_json

def test_parse_partial_json():
    test_cases = [
        ('{"a": "hello"}', {"a": "hello"}),
        ('{"a": "hell', {"a": "hell"}),
        ('{"a": ', {"a": None}),
        ('{"a": "hello", "b": "wor', {"a": "hello", "b": "wor"}),
        ('{"a": "hello", "b": ', {"a": "hello", "b": None}),
        ('{"key_with_colon: ": "val', {"key_with_colon: ": "val"}),
        ('{"a": "hello", "b": "wo', {"a": "hello", "b": "wo"}),
    ]

    for s, expected in test_cases:
        result = parse_partial_json(s)
        assert result == expected

def test_parse_partial_json_malformed():
    with pytest.raises(ValueError, match="Malformed partial JSON encountered."):
        parse_partial_json('{"a": "hello" "b": "world"}')

def test_parse_partial_json_array():
    assert parse_partial_json('["a", "b') == ["a", "b"]
    assert parse_partial_json('["a", ') == ["a"]
    assert parse_partial_json('[{"a": "hell') == [{"a": "hell"}]
