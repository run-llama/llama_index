"""Unit tests for the helpers that do not require a running FalkorDB."""

import pytest

from llama_index.graph_stores.falkordb.falkordb_property_graph import (
    FalkorDBPropertyGraphStore,
    _batched,
    convert_operator,
    escape_identifier,
    remove_empty_values,
    sample_value_type,
    to_plain_value,
)


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("PERSON", "`PERSON`"),
        ("Business Unit", "`Business Unit`"),
        ("A` REMOVE e:`__Entity__", "`A REMOVE e:__Entity__`"),
        ("`", "``"),
    ],
)
def test_escape_identifier(identifier: str, expected: str) -> None:
    assert escape_identifier(identifier) == expected


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("==", "="),
        ("!=", "<>"),
        ("in", "IN"),
        ("nin", "IN"),
        ("contains", "CONTAINS"),
        (">=", ">="),
        ("<", "<"),
    ],
)
def test_convert_operator(operator: str, expected: str) -> None:
    assert convert_operator(operator) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "BOOLEAN"),
        (3, "INTEGER"),
        (1.5, "FLOAT"),
        ("text", "STRING"),
        ([1, 2], "LIST"),
        ({"a": 1}, "MAP"),
        (None, "UNKNOWN"),
    ],
)
def test_sample_value_type(value: object, expected: str) -> None:
    assert sample_value_type(value) == expected


def test_to_plain_value_recurses() -> None:
    assert to_plain_value({"a": [1, {"b": 2}]}) == {"a": [1, {"b": 2}]}
    assert to_plain_value((1, 2)) == [1, 2]
    assert to_plain_value("text") == "text"


def test_to_plain_value_converts_driver_objects() -> None:
    from falkordb import Edge, Node

    node = Node(node_id=1, alias="n", labels="PERSON", properties={"name": "Alice"})
    edge = Edge(src_node=1, relation="KNOWS", dest_node=2, properties={"since": 2023})

    assert to_plain_value(node) == {"name": "Alice"}
    assert to_plain_value(edge) == {"since": 2023}
    assert to_plain_value([node, edge]) == [{"name": "Alice"}, {"since": 2023}]


def test_batched() -> None:
    rows = [{"i": i} for i in range(5)]
    assert list(_batched(rows, size=2)) == [
        [{"i": 0}, {"i": 1}],
        [{"i": 2}, {"i": 3}],
        [{"i": 4}],
    ]
    assert list(_batched([], size=2)) == []


def test_remove_empty_values() -> None:
    assert remove_empty_values({"a": 1, "b": None, "c": "", "d": []}) == {"a": 1}


def test_format_properties_handles_both_schema_shapes() -> None:
    formatted = FalkorDBPropertyGraphStore._format_properties(
        [{"property": "age", "sample": 30}, "name"]
    )
    assert formatted == [
        {"property": "age", "type": "INTEGER"},
        {"property": "name", "type": "UNKNOWN"},
    ]
    assert FalkorDBPropertyGraphStore._format_properties(None) == []
