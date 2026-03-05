from llama_index.core.llms.utils import parse_partial_json


def test_parse_partial_json_keeps_incomplete_string_value() -> None:
    """Regression test for #20541: unfinished string values should be preserved."""
    parsed = parse_partial_json('{"key": "value')

    assert parsed == {"key": "value"}


def test_parse_partial_json_drops_incomplete_object_key() -> None:
    """Unfinished object keys should still be removed to keep partial JSON valid."""
    parsed = parse_partial_json('{"key": 1, "dangling')

    assert parsed == {"key": 1}


def test_parse_partial_json_keeps_incomplete_string_value_in_array() -> None:
    """Unfinished string values in arrays should be preserved too."""
    parsed = parse_partial_json('{"items": ["alpha", "bet')

    assert parsed == {"items": ["alpha", "bet"]}
