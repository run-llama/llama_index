"""Tests for the SQL response parsers."""

import pytest

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.indices.struct_store.sql_retriever import (
    DefaultSQLParser,
    PGVectorSQLParser,
)
from llama_index.core.schema import QueryBundle


@pytest.fixture()
def query_bundle() -> QueryBundle:
    return QueryBundle(query_str="how many users are there?")


@pytest.mark.parametrize(
    "response",
    [
        "SELECT * FROM users",
        "```sql\nSELECT * FROM users\n```",
        "SQLQuery: SELECT * FROM users",
        "SELECT * FROM users\nSQLResult: 42",
    ],
)
def test_default_parser_keeps_trailing_identifier(
    response: str, query_bundle: QueryBundle
) -> None:
    """A table name ending in one of the fence characters must survive."""
    assert DefaultSQLParser().parse_response_to_sql(response, query_bundle) == (
        "SELECT * FROM users"
    )


@pytest.mark.parametrize(
    "response",
    [
        "SELECT * FROM users",
        "```sql\nSELECT * FROM users\n```",
        "SQLQuery: SELECT * FROM users",
        "SELECT * FROM users\nSQLResult: 42",
    ],
)
def test_pgvector_parser_keeps_trailing_identifier(
    response: str, query_bundle: QueryBundle
) -> None:
    """`str.strip("```sql")` used to eat the trailing "s" of "users"."""
    parser = PGVectorSQLParser(embed_model=MockEmbedding(embed_dim=1))
    assert parser.parse_response_to_sql(response, query_bundle) == (
        "SELECT * FROM users"
    )


def test_pgvector_parser_substitutes_query_vector(query_bundle: QueryBundle) -> None:
    parser = PGVectorSQLParser(embed_model=MockEmbedding(embed_dim=1))
    parsed = parser.parse_response_to_sql(
        "SELECT * FROM logs ORDER BY embedding <-> '[query_vector]'", query_bundle
    )
    assert "[query_vector]" not in parsed
    assert parsed.startswith("SELECT * FROM logs")
