"""
parse_partial_json must repair truncated JSON literals, not just strings
and structures.
"""

import pytest

from llama_index.core.llms.utils import parse_partial_json


def test_complete_json_unchanged():
    assert parse_partial_json('{"a": [1, 2], "b": true}') == {"a": [1, 2], "b": True}


def test_truncated_true_literal_completes():
    assert parse_partial_json('{"a": [1,2], "b": tru') == {"a": [1, 2], "b": True}


def test_truncated_false_literal_completes():
    assert parse_partial_json('{"a": [1,2], "b": fals') == {"a": [1, 2], "b": False}


def test_truncated_null_literal_completes():
    assert parse_partial_json('{"a": [1,2], "b": nu') == {"a": [1, 2], "b": None}
    assert parse_partial_json('{"a": [1,2], "b": nul') == {"a": [1, 2], "b": None}


def test_truncated_string_value_yields_none():
    assert parse_partial_json('{"a": "tru') == {"a": None}


def test_literal_inside_string_is_untouched():
    assert parse_partial_json('{"a": "a true story"}') == {"a": "a true story"}


def test_truncated_nested_structure_still_works():
    assert parse_partial_json('{"a": {"b": [1, 2') == {"a": {"b": [1, 2]}}


def test_malformed_still_raises():
    with pytest.raises(ValueError):
        parse_partial_json('{"a": [1,2], "b": tru,,}')


def test_mid_object_truncated_literal_completes():
    """A cut-off can leave the fragment mid-object: '{"a": tru, "b": 1}'."""
    assert parse_partial_json('{"a": tru, "b": 1}') == {"a": True, "b": 1}
    assert parse_partial_json('{"a": fals, "b": 1}') == {"a": False, "b": 1}


def test_quoted_fragment_stays_a_string():
    assert parse_partial_json('{"a": 1, "b": "tru"}') == {"a": 1, "b": "tru"}


def test_complete_literal_array_unchanged():
    assert parse_partial_json('{"a": [true, false]}') == {"a": [True, False]}
