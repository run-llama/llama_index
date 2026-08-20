from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llama_index.vector_stores.oracledb import base
from llama_index.vector_stores.oracledb import hybrid
from llama_index.vector_stores.oracledb import text as text_module
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


def test_create_hnsw_index_fills_missing_parameter_defaults() -> None:
    connection = MagicMock()
    cursor_context = MagicMock()
    cursor = MagicMock()
    cursor_context.__enter__.return_value = cursor
    connection.cursor.return_value = cursor_context

    with patch.object(base, "_index_exists", return_value=False):
        base._create_hnsw_index(
            connection,
            "docs",
            DistanceStrategy.DOT_PRODUCT,
            {"idx_name": "idx1", "neighbors": 64, "parallel": 2},
        )

    ddl = cursor.execute.call_args.args[0]
    assert "DISTANCE DOT" in ddl
    assert "parameters (type HNSW, neighbors 64, efConstruction 200)" in ddl
    assert " parallel 2" in ddl


def test_create_ivf_index_quotes_names_and_builds_parameters() -> None:
    connection = MagicMock()
    cursor_context = MagicMock()
    cursor = MagicMock()
    cursor_context.__enter__.return_value = cursor
    connection.cursor.return_value = cursor_context

    with patch.object(base, "_index_exists", return_value=False):
        base._create_ivf_index(
            connection,
            "docs",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            {
                "idx_name": "idx1",
                "neighbor_part": 16,
                "accuracy": 80,
                "parallel": 3,
            },
        )

    ddl = cursor.execute.call_args.args[0]
    assert 'CREATE VECTOR INDEX "IDX1" ON "DOCS"(embedding)' in ddl
    assert "WITH TARGET ACCURACY 80" in ddl
    assert "DISTANCE EUCLIDEAN" in ddl
    assert "PARAMETERS (type IVF, neighbor partitions 16)" in ddl
    assert "PARALLEL 3" in ddl


def test_vector_index_validation_rejects_bad_numeric_bounds_and_type() -> None:
    with pytest.raises(ValueError, match="accuracy must be at most 100"):
        base._create_hnsw_index(
            MagicMock(),
            "docs",
            DistanceStrategy.COSINE,
            {"idx_name": "idx1", "idx_type": "HNSW", "accuracy": 101},
        )

    with pytest.raises(ValueError, match="neighbors must be at least 2"):
        base._create_hnsw_index(
            MagicMock(),
            "docs",
            DistanceStrategy.COSINE,
            {"idx_name": "idx1", "idx_type": "HNSW", "neighbors": 1},
        )

    with pytest.raises(ValueError, match="idx_type must be IVF"):
        base._create_ivf_index(
            MagicMock(),
            "docs",
            DistanceStrategy.COSINE,
            {"idx_name": "idx1", "idx_type": "HNSW"},
        )


def test_create_config_uses_defaults_when_params_are_omitted() -> None:
    assert base._create_config({"idx_name": "IDX", "parallel": 1}, None) == {
        "idx_name": "IDX",
        "parallel": 1,
    }


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


def test_output_type_string_handler_maps_lob_types() -> None:
    cursor = MagicMock(arraysize=25)

    assert (
        base.output_type_string_handler(
            cursor, SimpleNamespace(type_code=base.oracledb.DB_TYPE_CLOB)
        )
        == cursor.var.return_value
    )
    cursor.var.assert_called_with(base.oracledb.DB_TYPE_LONG, arraysize=25)

    cursor.var.reset_mock()
    assert (
        base.output_type_string_handler(
            cursor, SimpleNamespace(type_code=base.oracledb.DB_TYPE_NCLOB)
        )
        == cursor.var.return_value
    )
    cursor.var.assert_called_with(base.oracledb.DB_TYPE_LONG_NVARCHAR, arraysize=25)

    assert (
        base.output_type_string_handler(cursor, SimpleNamespace(type_code=object()))
        is None
    )


def test_generate_accum_query_quotes_terms_and_supports_fuzzy() -> None:
    assert base._generate_accum_query("refund, policy") == '"refund" ACCUM "policy"'
    assert (
        base._generate_accum_query("refund policy", fuzzy=True)
        == 'fuzzy("refund") ACCUM fuzzy("policy")'
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


def test_hybrid_preference_parameters_infer_database_model() -> None:
    class FakeOracleEmbeddings:
        def __init__(self) -> None:
            self._params = {"provider": "database", "model": "ALL_MINILM_L12_V2"}

    with patch.object(hybrid, "OracleEmbeddings", FakeOracleEmbeddings):
        preference = hybrid.OracleVectorizerPreference.__new__(
            hybrid.OracleVectorizerPreference
        )
        preference.params = None
        preference.embeddings = FakeOracleEmbeddings()

        assert preference._get_preference_parameters() == {"model": "ALL_MINILM_L12_V2"}


def test_hybrid_preference_parameters_validate_explicit_configs() -> None:
    class FakeOracleEmbeddings:
        def __init__(self, params: dict) -> None:
            self._params = params

    database_embeddings = FakeOracleEmbeddings(
        {"provider": "database", "model": "ALL_MINILM_L12_V2"}
    )
    external_embeddings = FakeOracleEmbeddings(
        {"provider": "ocigenai", "credential_name": "CRED", "model": "cohere"}
    )

    with patch.object(hybrid, "OracleEmbeddings", FakeOracleEmbeddings):
        assert hybrid._validate_parameters(
            database_embeddings, {"model": "ALL_MINILM_L12_V2"}
        )
        assert hybrid._validate_parameters(
            external_embeddings, {"embedder_spec": external_embeddings._params}
        )

        with pytest.raises(ValueError, match="Mismatch between embedding"):
            hybrid._validate_parameters(database_embeddings, {"model": "OTHER"})

        with pytest.raises(ValueError, match="embedder_spec must exactly match"):
            hybrid._validate_parameters(
                external_embeddings,
                {"embedder_spec": {"provider": "ocigenai", "model": "other"}},
            )


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


def test_create_text_index_creates_when_missing_and_skips_when_present() -> None:
    connection = MagicMock()
    cursor_context = MagicMock()
    cursor = MagicMock()
    cursor_context.__enter__.return_value = cursor
    connection.cursor.return_value = cursor_context
    vector_store = SimpleNamespace(_quoted_table_name='"DOCS"')

    with (
        patch.object(text_module, "_get_connection") as get_connection,
        patch.object(text_module, "_index_exists", return_value=False),
    ):
        get_connection.return_value.__enter__.return_value = connection
        text_module.create_text_index(MagicMock(), "idx1", vector_store)

    cursor.execute.assert_called_once_with('CREATE SEARCH INDEX "IDX1" ON "DOCS"(text)')

    cursor.execute.reset_mock()
    with (
        patch.object(text_module, "_get_connection") as get_connection,
        patch.object(text_module, "_index_exists", return_value=True),
    ):
        get_connection.return_value.__enter__.return_value = connection
        text_module.create_text_index(MagicMock(), "idx1", vector_store)

    cursor.execute.assert_not_called()
