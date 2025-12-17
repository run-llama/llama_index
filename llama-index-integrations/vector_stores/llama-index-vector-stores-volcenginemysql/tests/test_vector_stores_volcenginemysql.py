"""
Tests for `VolcengineMySQLVectorStore`.

These tests are intentionally written as "SQL-construction" unit tests:

- No real database is contacted.
- We patch `sqlalchemy.create_engine()` and assert that the resulting SQLAlchemy
  statements contain the expected clauses.

This keeps the suite deterministic and runnable in CI without provisioning a
Volcengine MySQL instance.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.volcengine_mysql import VolcengineMySQLVectorStore


# A syntactically-valid connection string used across unit tests. We never
# connect to the target; `sqlalchemy.create_engine` is always patched.
_TEST_CONN_STR = "mysql+pymysql://user:pass@localhost:3306/db"


def test_vector_store_reports_class_name() -> None:
    """`class_name()` is used for registry / serialization roundtrips."""
    assert VolcengineMySQLVectorStore.class_name() == "VolcengineMySQLVectorStore"


def test_client_property_tracks_engine_lifecycle() -> None:
    """`client` should be non-None after initialization and cleared on close()."""
    with patch("sqlalchemy.create_engine") as mock_create_engine:
        with patch.object(VolcengineMySQLVectorStore, "_validate_server_capability"):
            with patch.object(
                VolcengineMySQLVectorStore, "_create_table_if_not_exists"
            ):
                store = VolcengineMySQLVectorStore(
                    connection_string=_TEST_CONN_STR,
                    perform_setup=True,
                )

                # `perform_setup=True` triggers `_initialize()`, so an engine gets created.
                assert store.client is not None

                store.close()
                assert store.client is None


def test_constructor_with_setup_runs_validation_and_table_creation() -> None:
    """When setup is requested, the store validates server features and creates tables."""
    with patch("sqlalchemy.create_engine") as mock_create_engine:
        with patch.object(
            VolcengineMySQLVectorStore, "_validate_server_capability"
        ) as mock_validate:
            with patch.object(
                VolcengineMySQLVectorStore, "_create_table_if_not_exists"
            ) as mock_create_table:
                store = VolcengineMySQLVectorStore(
                    connection_string=_TEST_CONN_STR,
                    perform_setup=True,
                )

                mock_create_engine.assert_called_once()
                mock_validate.assert_called_once()
                mock_create_table.assert_called_once()
                assert store._is_initialized is True


def test_constructor_without_setup_skips_validation_and_ddl() -> None:
    """When setup is disabled, we still create an engine but skip DDL and validation."""
    with patch("sqlalchemy.create_engine") as mock_create_engine:
        with patch.object(
            VolcengineMySQLVectorStore, "_validate_server_capability"
        ) as mock_validate:
            with patch.object(
                VolcengineMySQLVectorStore, "_create_table_if_not_exists"
            ) as mock_create_table:
                store = VolcengineMySQLVectorStore(
                    connection_string=_TEST_CONN_STR,
                    perform_setup=False,
                )

                # Connection is established but setup steps are skipped
                mock_create_engine.assert_called_once()
                mock_validate.assert_not_called()
                mock_create_table.assert_not_called()
                assert store._is_initialized is True


def test_from_params_builds_connection_string_and_forwards_kwargs() -> None:
    """`from_params()` should synthesize a MySQL connection string and pass through options."""
    with patch.object(
        VolcengineMySQLVectorStore, "__init__", return_value=None
    ) as mock_init:
        VolcengineMySQLVectorStore.from_params(
            host="localhost",
            port=3306,
            user="user",
            password="password",
            database="db",
            table_name="custom_table",
        )

        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert (
            "mysql+pymysql://user:password@localhost:3306/db"
            in call_kwargs["connection_string"]
        )
        assert call_kwargs["table_name"] == "custom_table"


def test_validate_server_capability_accepts_enabled_flag_and_rejects_disabled() -> None:
    """The feature flag `loose_vector_index_enabled` must be ON for vector search."""
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_result = MagicMock()

    # Successful case
    mock_result.fetchone.return_value = ["loose_vector_index_enabled", "ON"]
    mock_connection.execute.return_value = mock_result
    mock_engine.connect.return_value.__enter__.return_value = mock_connection

    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        store = VolcengineMySQLVectorStore(
            connection_string=_TEST_CONN_STR,
            perform_setup=False,
        )
        store._validate_server_capability()

    # Failure case
    mock_result.fetchone.return_value = ["loose_vector_index_enabled", "OFF"]
    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        store = VolcengineMySQLVectorStore(
            connection_string=_TEST_CONN_STR,
            perform_setup=False,
        )
        with pytest.raises(
            ValueError, match="Volcengine MySQL vector index is not enabled"
        ):
            store._validate_server_capability()


def test_create_table_if_missing_emits_vector_index_ddl() -> None:
    """The DDL should include a vector column and the corresponding VECTOR INDEX."""
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_connection

    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        store = VolcengineMySQLVectorStore(
            connection_string=_TEST_CONN_STR,
            table_name="test_table",
            perform_setup=False,
        )
        store._create_table_if_not_exists()

        mock_connection.execute.assert_called_once()
        stmt = mock_connection.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS `test_table`" in str(stmt)
        assert "VECTOR INDEX idx_embedding (embedding)" in str(stmt)


def test_add_inserts_rows_and_returns_node_ids() -> None:
    """`add()` should batch-insert rows and return the IDs in the same order."""
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_connection

    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        store = VolcengineMySQLVectorStore(
            connection_string=_TEST_CONN_STR,
            perform_setup=False,
        )

        # In production these are produced by an index / embedding model. In the unit
        # test we provide fixed vectors so we can assert on the SQL payload.
        nodes = [
            TextNode(
                text="test text 1",
                id_="id1",
                embedding=[1.0, 2.0],
                metadata={"key": "val1"},
            ),
            TextNode(
                text="test text 2",
                id_="id2",
                embedding=[3.0, 4.0],
                metadata={"key": "val2"},
            ),
        ]

        ids = store.add(nodes)

        assert ids == ["id1", "id2"]
        mock_connection.execute.assert_called_once()
        # Verify call args
        args = mock_connection.execute.call_args
        assert "INSERT INTO `llamaindex`" in str(args[0][0])
        assert len(args[0][1]) == 2  # 2 rows


def test_delete_by_ref_doc_id_builds_expected_where_clause() -> None:
    """`delete()` removes rows by `ref_doc_id` stored inside the JSON metadata."""
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_connection

    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        store = VolcengineMySQLVectorStore(
            connection_string=_TEST_CONN_STR,
            perform_setup=False,
        )

        store.delete(ref_doc_id="ref_id_1")

        mock_connection.execute.assert_called_once()
        stmt = mock_connection.execute.call_args[0][0]
        assert "DELETE FROM `llamaindex`" in str(stmt)
        assert "JSON_EXTRACT(metadata, '$.ref_doc_id') = :doc_id" in str(stmt)


def test_delete_nodes_uses_in_clause_for_node_ids() -> None:
    """`delete_nodes()` should translate a list of IDs into a SQL IN clause."""
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_connection

    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        store = VolcengineMySQLVectorStore(
            connection_string=_TEST_CONN_STR,
            perform_setup=False,
        )

        store.delete_nodes(node_ids=["id1", "id2"])

        mock_connection.execute.assert_called_once()
        stmt = mock_connection.execute.call_args[0][0]
        assert "DELETE FROM `llamaindex`" in str(stmt)
        # Note: SQLAlchemy expands IN parameters into a special
        # "__[POSTCOMPILE_...]" placeholder. We assert on the existence of the
        # IN clause rather than matching the parameter token.
        assert "WHERE node_id IN" in str(stmt)


def test_query_returns_scored_nodes_from_distance() -> None:
    """`query()` should return nodes + similarities derived from the distance metric."""
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_result = MagicMock()

    # The vector store reads columns from SQLAlchemy result rows by attribute.
    # We model a single returned row to keep the unit test focused.
    row = Mock()
    row.node_id = "id1"
    row.text = "text1"
    row.metadata = '{"key": "val"}'
    row.distance = 0.1

    mock_result.__iter__.return_value = [row]
    mock_connection.execute.return_value = mock_result
    mock_engine.connect.return_value.__enter__.return_value = mock_connection

    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        store = VolcengineMySQLVectorStore(
            connection_string=_TEST_CONN_STR,
            perform_setup=False,
        )

        query = VectorStoreQuery(
            query_embedding=[1.0, 2.0],
            similarity_top_k=2,
        )

        result = store.query(query)

        assert len(result.nodes) == 1
        assert result.nodes[0].node_id == "id1"
        # Implementation detail: similarity is derived from distance.
        # Similarity = 1 / (1 + distance) = 1 / 1.1 ~= 0.909
        assert pytest.approx(result.similarities[0], 0.001) == 0.909

        mock_connection.execute.assert_called()
        # Verify query structure
        stmt = str(mock_connection.execute.call_args_list[-1][0][0])
        assert "SELECT" in stmt
        assert "L2_DISTANCE" in stmt  # Default is l2
        assert "ORDER BY distance LIMIT :limit" in stmt


def test_query_applies_metadata_filters_as_json_extract_predicates() -> None:
    """Metadata filters are stored in a JSON column and translated into predicates."""
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_result = MagicMock()
    mock_result.__iter__.return_value = []
    mock_connection.execute.return_value = mock_result
    mock_engine.connect.return_value.__enter__.return_value = mock_connection

    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        store = VolcengineMySQLVectorStore(
            connection_string=_TEST_CONN_STR,
            perform_setup=False,
        )

        # Use a mix of string and numeric filters to exercise both quoting and
        # numeric comparison paths.
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="key1", value="val1", operator=FilterOperator.EQ),
                MetadataFilter(key="key2", value=10, operator=FilterOperator.GT),
            ]
        )

        query = VectorStoreQuery(
            query_embedding=[1.0, 2.0],
            filters=filters,
        )

        store.query(query)

        stmt = str(mock_connection.execute.call_args_list[-1][0][0])
        assert "WHERE" in stmt
        assert (
            "JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.key1')) = :filter_param_0" in stmt
        )
        assert "JSON_EXTRACT(metadata, '$.key2') > :filter_param_1" in stmt


def test_get_nodes_fetches_text_and_metadata_for_ids() -> None:
    """`get_nodes()` should fetch the stored payload for the requested IDs."""
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_result = MagicMock()

    row = Mock()
    row.text = "text1"
    row.metadata = '{"key": "val"}'

    mock_result.__iter__.return_value = [row]
    mock_connection.execute.return_value = mock_result
    mock_engine.connect.return_value.__enter__.return_value = mock_connection

    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        store = VolcengineMySQLVectorStore(
            connection_string=_TEST_CONN_STR,
            perform_setup=False,
        )

        nodes = store.get_nodes(node_ids=["id1"])

        assert len(nodes) == 1
        assert nodes[0].text == "text1"

        stmt = str(mock_connection.execute.call_args[0][0])
        # Similar to delete_nodes(), SQLAlchemy rewrites the IN parameter
        # placeholder, so we avoid matching the exact token.
        assert "SELECT text, metadata FROM `llamaindex`" in stmt
        assert "WHERE node_id IN" in stmt


def test_clear_deletes_all_rows_from_table() -> None:
    """`clear()` is a convenience that wipes all entries (but keeps the table)."""
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_connection

    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        store = VolcengineMySQLVectorStore(
            connection_string=_TEST_CONN_STR,
            perform_setup=False,
        )

        store.clear()

        stmt = str(mock_connection.execute.call_args[0][0])
        assert "DELETE FROM `llamaindex`" in stmt


def test_drop_removes_table_and_disposes_engine() -> None:
    """`drop()` should issue DROP TABLE and release engine resources."""
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_connection

    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        store = VolcengineMySQLVectorStore(
            connection_string=_TEST_CONN_STR,
            perform_setup=False,
        )

        store.drop()

        stmt = str(mock_connection.execute.call_args[0][0])
        assert "DROP TABLE IF EXISTS `llamaindex`" in stmt

        # Verify engine was disposed
        mock_engine.dispose.assert_called_once()
