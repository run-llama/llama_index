"""Helpers for embedding identifiers in Cypher queries safely."""

from typing import Any


def escape_identifier(value: str) -> str:
    """
    Quote ``value`` for use where Cypher expects an identifier.

    Node labels and relationship types cannot be supplied as query parameters,
    so they have to be interpolated into the query text. Cypher's escaping rule
    for a quoted identifier is to wrap it in backticks and double any backtick
    it contains; doing that keeps legitimate values such as ``has space`` or
    ``is-a`` working while making it impossible to terminate the identifier and
    append further clauses.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"Cypher identifier must be a string, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("Cypher identifier must not be empty")
    if "\x00" in value:
        raise ValueError("Cypher identifier must not contain a null byte")
    return "`" + value.replace("`", "``") + "`"


def escape_int(value: Any, name: str) -> int:
    """
    Coerce ``value`` to an ``int`` for interpolation into a Cypher query.

    Used for operands such as a variable-length path bound or a LIMIT, which
    cannot always be parameterised. Anything that is not integral is rejected
    rather than interpolated.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            value = int(str(value))
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


def escape_string_literal(value: str) -> str:
    """
    Escape ``value`` for use inside a single-quoted Cypher string literal.

    Preferred practice is to pass such values as query parameters; this exists
    for the few call sites where the value is embedded in a procedure argument
    that is assembled as query text.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"Cypher string literal must be a string, got {type(value).__name__}"
        )
    return value.replace("\\", "\\\\").replace("'", "\\'")
