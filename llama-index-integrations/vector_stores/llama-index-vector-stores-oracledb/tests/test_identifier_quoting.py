from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llama_index.vector_stores.oracledb import base
from llama_index.vector_stores.oracledb import OraLlamaVS
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.oracledb.base import DistanceStrategy
from llama_index.vector_stores.oracledb.hybrid import _get_hybrid_index_ddl
from llama_index.vector_stores.oracledb.text import _get_text_index_ddl


def test_quote_identifier_quotes_unquoted_sql_tokens_as_identifier() -> None:
    assert (
        base._quote_identifier("TARGET_TABLE WHERE 1=1 --")
        == '"TARGET_TABLE WHERE 1=1 --"'
    )
    assert base._quote_identifier("IDX1 NOLOGGING") == '"IDX1 NOLOGGING"'
    assert base._quote_identifier("idx on dual") == '"IDX ON DUAL"'
    assert base._quote_identifier("DOCS; DROP TABLE DOCS") == (
        '"DOCS; DROP TABLE DOCS"'
    )


def test_quote_identifier_rejects_malformed_identifiers() -> None:
    for name in ["", ".", "owner.", '"unterminated', 'bad"name']:
        with pytest.raises(ValueError):
            base._quote_identifier(name)


def test_quote_identifier_uses_oracle_unquoted_case_semantics() -> None:
    assert base._quote_identifier("docs") == '"DOCS"'
    assert base._quote_identifier("owner.docs") == '"OWNER"."DOCS"'
    assert base._quote_identifier('"Mixed"."Case"') == '"Mixed"."Case"'


def test_index_exists_uses_exact_quoted_lookup_name() -> None:
    connection = MagicMock()
    cursor_context = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = ("IDX1",)
    cursor_context.__enter__.return_value = cursor
    connection.cursor.return_value = cursor_context

    assert base._index_exists(connection, "idx1") is True

    query = cursor.execute.call_args.args[0]
    assert "upper(" not in query.lower()
    assert cursor.execute.call_args.kwargs == {"idx_name": "IDX1"}


def test_index_exists_uses_owner_for_schema_qualified_name() -> None:
    connection = MagicMock()
    cursor_context = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = ("IDX1",)
    cursor_context.__enter__.return_value = cursor
    connection.cursor.return_value = cursor_context

    assert base._index_exists(connection, "schema.idx1") is True

    query = cursor.execute.call_args.args[0]
    assert "owner = :owner" in query
    assert cursor.execute.call_args.kwargs == {"idx_name": "IDX1", "owner": "SCHEMA"}


def test_index_exists_rejects_more_than_schema_qualified_name() -> None:
    with pytest.raises(ValueError, match="schema-qualified"):
        base._index_exists(MagicMock(), "one.two.three")


def test_create_hnsw_index_rejects_injected_parallel() -> None:
    with pytest.raises(ValueError):
        base._create_hnsw_index(
            MagicMock(),
            "docs",
            DistanceStrategy.COSINE,
            {
                "idx_name": "idx1",
                "idx_type": "HNSW",
                "parallel": "1 NOLOGGING",
            },
        )


def test_create_hnsw_index_quotes_table_and_index_names() -> None:
    connection = MagicMock()
    cursor_context = MagicMock()
    cursor = MagicMock()
    cursor_context.__enter__.return_value = cursor
    connection.cursor.return_value = cursor_context

    with patch.object(base, "_index_exists", return_value=False):
        base._create_hnsw_index(
            connection,
            "docs",
            DistanceStrategy.COSINE,
            {
                "idx_name": "idx1",
                "idx_type": "HNSW",
                "parallel": 1,
            },
        )

    ddl = cursor.execute.call_args.args[0]
    assert 'create vector index "IDX1" on "DOCS"(embedding)' in ddl
    assert " parallel 1" in ddl


def test_drop_table_purge_quotes_injected_table_name() -> None:
    connection = MagicMock()
    cursor_context = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = (1,)
    cursor_context.__enter__.return_value = cursor
    connection.cursor.return_value = cursor_context

    with patch.object(base, "_get_connection") as get_connection:
        get_connection.return_value.__enter__.return_value = connection
        base.drop_table_purge(MagicMock(), "docs WHERE 1=1 --")

    assert cursor_context.execute.call_args.args[0] == (
        'DROP TABLE "DOCS WHERE 1=1 --" PURGE'
    )


def test_vector_store_builders_use_private_quoted_table_name() -> None:
    vector_store = OraLlamaVS.model_construct(
        table_name="docs",
        distance_strategy=DistanceStrategy.COSINE,
        batch_size=1,
        params=None,
    )
    object.__setattr__(vector_store, "_quoted_table_name", '"DOCS"')

    insert_sql, _ = vector_store._build_insert([])
    query_sql = vector_store._build_query("COSINE", 1)

    assert 'INSERT INTO "DOCS"' in insert_sql
    assert 'FROM "DOCS"' in query_sql


def test_hybrid_query_uses_json_bind_and_quoted_index_name() -> None:
    vector_store = OraLlamaVS.model_construct(
        table_name="docs",
        distance_strategy=DistanceStrategy.COSINE,
        batch_size=1,
        params=None,
        hybrid_index_name="HYB_deadbeef",
        hybrid_search_params=None,
        use_fuzzy_text_search=False,
    )
    object.__setattr__(vector_store, "_quoted_table_name", '"docs"')

    query = VectorStoreQuery(
        query_str="database",
        similarity_top_k=1,
        mode=VectorStoreQueryMode.HYBRID,
    )

    query_sql, params = vector_store._get_hybrid_query(query)

    assert query_sql == "SELECT DBMS_HYBRID_VECTOR.SEARCH(json(:search_params))"
    assert params["search_params"]["hybrid_index_name"] == '"HYB_DEADBEEF"'


def test_hybrid_index_ddl_rejects_injected_filter_and_order_columns() -> None:
    preference = SimpleNamespace(preference_name="pref")

    with pytest.raises(ValueError, match="filter_by contains an invalid identifier"):
        _get_hybrid_index_ddl(
            preference,
            "idx1",
            "docs",
            {"filter_by": ["text DESC"]},
        )

    with pytest.raises(ValueError, match="order_by contains an invalid identifier"):
        _get_hybrid_index_ddl(
            preference,
            "idx1",
            "docs",
            {"order_by": ["metadata NULLS FIRST"]},
        )


def test_hybrid_index_ddl_quotes_index_name_as_identifier() -> None:
    preference = SimpleNamespace(preference_name="pref")

    ddl = _get_hybrid_index_ddl(preference, "idx1 NOLOGGING", "docs", {})

    assert 'CREATE HYBRID VECTOR INDEX "IDX1 NOLOGGING" ON' in ddl


@pytest.mark.parametrize(
    "parameter_name",
    [
        " vectorizer",
        "vectorizer ",
        " MODEL ",
        "Embedder_Spec",
        " vector_idxtype ",
    ],
)
def test_hybrid_index_ddl_rejects_reserved_parameters_after_cleanup(
    parameter_name: str,
) -> None:
    preference = SimpleNamespace(preference_name="pref")

    with pytest.raises(ValueError, match="Vectorization parameters must be given"):
        _get_hybrid_index_ddl(
            preference,
            "idx1",
            "docs",
            {"parameters": {parameter_name: "ignored"}},
        )


def test_hybrid_index_ddl_quotes_identifiers_in_clauses() -> None:
    preference = SimpleNamespace(preference_name="pref")

    ddl = _get_hybrid_index_ddl(
        preference,
        "idx1",
        "docs",
        {"filter_by": ["doc_id"], "order_by": ["metadata"], "parallel": 1},
    )

    assert 'CREATE HYBRID VECTOR INDEX "IDX1" ON' in ddl
    assert '"DOCS"(text)' in ddl
    assert 'FILTER BY "DOC_ID"' in ddl
    assert 'ORDER BY "METADATA" ASC' in ddl
    assert "PARALLEL 1" in ddl


def test_hybrid_index_ddl_validates_clause_types() -> None:
    preference = SimpleNamespace(preference_name="pref")

    with pytest.raises(ValueError, match="filter_by must be a list of column names"):
        _get_hybrid_index_ddl(preference, "idx1", "docs", {"filter_by": "doc_id"})

    with pytest.raises(ValueError, match="order_by must contain only column names"):
        _get_hybrid_index_ddl(preference, "idx1", "docs", {"order_by": [1]})

    with pytest.raises(ValueError, match="order_by_asc must be a boolean"):
        _get_hybrid_index_ddl(
            preference, "idx1", "docs", {"order_by": ["doc_id"], "order_by_asc": "yes"}
        )

    for parallel in [0, -1, True, "1"]:
        with pytest.raises(ValueError, match="parallel must be a positive integer"):
            _get_hybrid_index_ddl(preference, "idx1", "docs", {"parallel": parallel})


def test_text_index_ddl_quotes_injected_index_name() -> None:
    vector_store = SimpleNamespace(_quoted_table_name='"DOCS"')

    assert _get_text_index_ddl("idx1 NOLOGGING", vector_store) == (
        'CREATE SEARCH INDEX "IDX1 NOLOGGING" ON "DOCS"(text)'
    )


def test_text_index_ddl_uses_quoted_table_name() -> None:
    vector_store = SimpleNamespace(_quoted_table_name='"DOCS"')

    assert (
        _get_text_index_ddl("idx1", vector_store)
        == 'CREATE SEARCH INDEX "IDX1" ON "DOCS"(text)'
    )
