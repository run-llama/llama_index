import pytest

from llama_index.core.llms.utils import parse_partial_json


def test_parse_partial_json_completes_truncated_true() -> None:
    assert parse_partial_json('{"a": [1, 2], "b": tru') == {"a": [1, 2], "b": True}


def test_parse_partial_json_completes_truncated_false() -> None:
    assert parse_partial_json('{"a": fal') == {"a": False}


def test_parse_partial_json_completes_truncated_null() -> None:
    assert parse_partial_json('{"a": nu') == {"a": None}
    assert parse_partial_json('{"a": nul') == {"a": None}


def test_parse_partial_json_keeps_arguments_before_truncated_literal() -> None:
    assert parse_partial_json('{"query": "test", "limit": tru') == {
        "query": "test",
        "limit": True,
    }


def test_parse_partial_json_leaves_complete_literals_alone() -> None:
    assert parse_partial_json('{"a": true}') == {"a": True}
    assert parse_partial_json('{"a": false, "b": null}') == {"a": False, "b": None}


def test_parse_partial_json_rejects_unknown_trailing_word() -> None:
    with pytest.raises(ValueError, match="Malformed partial JSON"):
        parse_partial_json('{"a": trux')


def test_parse_partial_json_completes_nested_truncated_literal() -> None:
    assert parse_partial_json('{"a": {"b": [1, 2, tru') == {"a": {"b": [1, 2, True]}}


def test_parse_partial_json_ignores_trailing_whitespace() -> None:
    assert parse_partial_json('{"a": tru   ') == {"a": True}


def test_parse_partial_json_still_defaults_incomplete_value_to_null() -> None:
    assert parse_partial_json('{"a":') == {"a": None}
