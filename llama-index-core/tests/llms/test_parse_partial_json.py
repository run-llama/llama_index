"""Tests for parse_partial_json in llama_index.core.llms.utils."""

import pytest

from llama_index.core.llms.utils import parse_partial_json


class TestParsePartialJsonCompleteCases:
    """Complete, well-formed JSON should round-trip unchanged."""

    def test_complete_simple_object(self):
        assert parse_partial_json('{"key": "value"}') == {"key": "value"}

    def test_complete_nested_object(self):
        assert parse_partial_json('{"a": {"b": 1}}') == {"a": {"b": 1}}

    def test_complete_object_multiple_keys(self):
        assert parse_partial_json('{"a": 1, "b": 2}') == {"a": 1, "b": 2}

    def test_complete_object_with_array(self):
        assert parse_partial_json('{"items": [1, 2, 3]}') == {"items": [1, 2, 3]}


class TestParsePartialJsonIncompleteValue:
    """Truncated string values should be preserved, not dropped (issue #20541)."""

    def test_incomplete_string_value_returns_partial_value(self):
        # Root cause of #20541: {"key": "value  →  was returning {'key': None}
        result = parse_partial_json('{"key": "value')
        assert result == {"key": "value"}

    def test_incomplete_string_value_empty_partial(self):
        # Opening quote for value with no content yet
        result = parse_partial_json('{"key": "')
        assert result == {"key": ""}

    def test_incomplete_string_value_second_key(self):
        # First key-value pair is complete; second value is truncated
        result = parse_partial_json('{"k1": "v1", "k2": "val')
        assert result == {"k1": "v1", "k2": "val"}

    def test_incomplete_string_value_with_spaces(self):
        result = parse_partial_json('{"description": "hello world')
        assert result == {"description": "hello world"}

    def test_incomplete_numeric_value_is_not_affected(self):
        # Numeric values are not strings; the function should handle truncated objects too
        result = parse_partial_json('{"count": 42')
        assert result == {"count": 42}


class TestParsePartialJsonIncompleteKey:
    """Truncated string keys (no colon yet) should be removed, not completed."""

    def test_incomplete_key_is_removed(self):
        # Only the object opener and a partial key — remove the partial key
        result = parse_partial_json('{"key')
        assert result == {}

    def test_incomplete_key_after_complete_pair(self):
        # First pair complete; second key is truncated
        result = parse_partial_json('{"a": 1, "b')
        assert result == {"a": 1}


class TestParsePartialJsonMiscellaneous:
    """Other partial-JSON edge cases."""

    def test_trailing_comma_removed(self):
        result = parse_partial_json('{"a": 1,')
        assert result == {"a": 1}

    def test_incomplete_object_with_colon_no_value(self):
        # Key present, colon present, but value missing entirely
        result = parse_partial_json('{"key":')
        assert result == {"key": None}

    def test_malformed_json_raises(self):
        with pytest.raises(ValueError, match="Malformed"):
            parse_partial_json('{"key": }')

    def test_empty_object(self):
        assert parse_partial_json("{}") == {}

    def test_empty_string_raises(self):
        with pytest.raises((ValueError, Exception)):
            parse_partial_json("")
