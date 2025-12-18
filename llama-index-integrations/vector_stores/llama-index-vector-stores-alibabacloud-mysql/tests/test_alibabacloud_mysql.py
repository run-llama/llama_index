"""Unit tests for AlibabaCloudMySQLVectorStore to improve test coverage."""

import json
from unittest.mock import Mock, patch, MagicMock

import pytest
from mysql.connector.errors import Error as MySQLError

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.alibabacloud_mysql import AlibabaCloudMySQLVectorStore


def test_class_name() -> None:
    """Test class_name method."""
    assert AlibabaCloudMySQLVectorStore.class_name() == "AlibabaCloudMySQLVectorStore"


def test_client_property() -> None:
    """Test client property."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
        with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
            )

            # Test when initialized
            store._is_initialized = True
            assert store.client is not None

            # Test when not initialized
            store._is_initialized = False
            assert store.client is None


def test_create_connection_pool() -> None:
    """Test _create_connection_pool method."""
    with patch("mysql.connector.pooling.MySQLConnectionPool") as mock_pool:
        with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                charset="utf8",
                max_connection=5,
            )

            # Verify the pool was created with correct parameters
            mock_pool.assert_called_once_with(
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                charset="utf8",
                autocommit=True,
                pool_name="pool_test_table",
                pool_size=5,
                pool_reset_session=True,
            )


def test_get_cursor_context_manager() -> None:
    """Test _get_cursor context manager."""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("mysql.connector.pooling.MySQLConnectionPool") as mock_pool:
        mock_pool.get_connection.return_value = mock_conn
        with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
            )

            # Test the context manager
            with patch.object(store, "_pool", mock_pool):
                with store._get_cursor() as cursor:
                    assert cursor == mock_cursor

                # Verify cleanup
                mock_cursor.close.assert_called_once()
                mock_conn.close.assert_called_once()


def test_check_vector_support_success() -> None:
    """Test _check_vector_support method with successful case."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = [
        {"vector_support": True},
        {"Variable_name": "rds_release_date", "Value": "20251031"}
    ]

    with patch.object(AlibabaCloudMySQLVectorStore, "_get_cursor") as mock_get_cursor:
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor
        with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
            with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                )

                # Should not raise any exception
                store._check_vector_support()


def test_check_vector_support_no_vector_functions() -> None:
    """Test _check_vector_support method when vector functions are not available."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {"vector_support": False}

    with patch.object(AlibabaCloudMySQLVectorStore, "_get_cursor") as mock_get_cursor:
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor
        with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
            with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                )

                with pytest.raises(ValueError, match="RDS MySQL Vector functions are not available"):
                    store._check_vector_support()


def test_check_vector_support_old_release_date() -> None:
    """Test _check_vector_support method with old release date."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = [
        {"vector_support": True},
        {"Variable_name": "rds_release_date", "Value": "20251030"}
    ]

    with patch.object(AlibabaCloudMySQLVectorStore, "_get_cursor") as mock_get_cursor:
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor
        with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
            with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                )

                with pytest.raises(ValueError, match="rds_release_date must be 20251031 or later"):
                    store._check_vector_support()


def test_check_vector_support_no_release_date_variable() -> None:
    """Test _check_vector_support method when rds_release_date variable is not available."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = [
        {"vector_support": True},
        None  # No rds_release_date variable
    ]

    with patch.object(AlibabaCloudMySQLVectorStore, "_get_cursor") as mock_get_cursor:
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor
        with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
            with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                )

                with pytest.raises(ValueError, match="Unable to retrieve rds_release_date variable"):
                    store._check_vector_support()


def test_check_vector_support_function_error() -> None:
    """Test _check_vector_support method when VEC_FromText function raises an error."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_get_cursor") as mock_get_cursor:
        mock_get_cursor.return_value.__enter__.side_effect = MySQLError("FUNCTION test.VEC_FromText does not exist")
        with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
            with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                )

                with pytest.raises(ValueError, match="RDS MySQL Vector functions are not available"):
                    store._check_vector_support()


def test_initialize_invalid_distance_method() -> None:
    """Test _initialize method with invalid distance method."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
        with patch.object(AlibabaCloudMySQLVectorStore, "_check_vector_support"):
            with pytest.raises(ValueError, match="Distance method 'INVALID' is not supported"):
                AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                    distance_method="INVALID"
                )


def test_initialize_success() -> None:
    """Test _initialize method success case."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
        with patch.object(AlibabaCloudMySQLVectorStore, "_check_vector_support") as mock_check:
            with patch.object(AlibabaCloudMySQLVectorStore, "_create_table_if_not_exists") as mock_create_table:
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                    perform_setup=True
                )

                # Verify methods were called
                mock_check.assert_called_once()
                mock_create_table.assert_called_once()
                assert store._is_initialized is True


def test_initialize_without_setup() -> None:
    """Test _initialize method when perform_setup is False."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
        with patch.object(AlibabaCloudMySQLVectorStore, "_check_vector_support") as mock_check:
            with patch.object(AlibabaCloudMySQLVectorStore, "_create_table_if_not_exists") as mock_create_table:
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                    perform_setup=False
                )

                # Verify methods were called
                mock_check.assert_called_once()
                mock_create_table.assert_not_called()
                assert store._is_initialized is True


def test_node_to_table_row() -> None:
    """Test _node_to_table_row method."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
        with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
            )

            # Create a test node
            node = TextNode(
                text="test text",
                id_="test-id",
                metadata={"key": "value"},
                embedding=[1.0, 2.0, 3.0]
            )

            # Convert node to table row
            row = store._node_to_table_row(node)

            assert row["node_id"] == "test-id"
            assert row["text"] == "test text"
            assert row["embedding"] == [1.0, 2.0, 3.0]
            assert "key" in row["metadata"]


def test_to_mysql_operator() -> None:
    """Test _to_mysql_operator method."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
        with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
            )

            # Test all supported operators
            assert store._to_mysql_operator(FilterOperator.EQ) == "="
            assert store._to_mysql_operator(FilterOperator.GT) == ">"
            assert store._to_mysql_operator(FilterOperator.LT) == "<"
            assert store._to_mysql_operator(FilterOperator.NE) == "!="
            assert store._to_mysql_operator(FilterOperator.GTE) == ">="
            assert store._to_mysql_operator(FilterOperator.LTE) == "<="
            assert store._to_mysql_operator(FilterOperator.IN) == "IN"
            assert store._to_mysql_operator(FilterOperator.NIN) == "NOT IN"

            # Test unsupported operator (should fallback to =)
            with patch("llama_index.vector_stores.alibabacloud_mysql.base._logger") as mock_logger:
                assert store._to_mysql_operator("UNSUPPORTED") == "="
                mock_logger.warning.assert_called_once()


def test_build_filter_clause() -> None:
    """Test _build_filter_clause method."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
        with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
            )

            # Test simple equality filter
            filter_eq = MetadataFilter(key="category", value="test", operator=FilterOperator.EQ)
            clause, values = store._build_filter_clause(filter_eq)
            assert "JSON_VALUE(metadata, '$.category') =" in clause
            assert values == ["test"]

            # Test IN filter
            filter_in = MetadataFilter(key="category", value=["test1", "test2"], operator=FilterOperator.IN)
            clause, values = store._build_filter_clause(filter_in)
            assert "JSON_VALUE(metadata, '$.category') IN" in clause
            assert values == ["test1", "test2"]


def test_filters_to_where_clause() -> None:
    """Test _filters_to_where_clause method."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
        with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
            )

            # Test simple filters
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="category", value="test", operator=FilterOperator.EQ),
                    MetadataFilter(key="priority", value=1, operator=FilterOperator.GT),
                ],
                condition="and"
            )

            clause, values = store._filters_to_where_clause(filters)
            assert "JSON_VALUE(metadata, '$.category') =" in clause
            assert "JSON_VALUE(metadata, '$.priority') >" in clause
            assert "AND" in clause
            assert values == ["test", 1]

            # Test OR condition with multiple filters
            filters_or = MetadataFilters(
                filters=[
                    MetadataFilter(key="category", value="test", operator=FilterOperator.EQ),
                    MetadataFilter(key="type", value="document", operator=FilterOperator.EQ),
                ],
                condition="or"
            )

            clause, values = store._filters_to_where_clause(filters_or)
            assert "OR" in clause


def test_db_rows_to_query_result() -> None:
    """Test _db_rows_to_query_result method."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
        with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
            )

            # Create test rows
            from llama_index.vector_stores.alibabacloud_mysql.base import DBEmbeddingRow
            test_rows = [
                DBEmbeddingRow(
                    node_id="test-id-1",
                    text="test text 1",
                    metadata={"_node_content": "{\"id_\": \"test-id-1\", \"text\": \"test text 1\"}"},
                    similarity=0.9
                ),
                DBEmbeddingRow(
                    node_id="test-id-2",
                    text="test text 2",
                    metadata={"_node_content": "{\"id_\": \"test-id-2\", \"text\": \"test text 2\"}"},
                    similarity=0.8
                )
            ]

            # Convert to query result
            result = store._db_rows_to_query_result(test_rows)

            assert len(result.nodes) == 2
            assert len(result.similarities) == 2
            assert len(result.ids) == 2
            assert result.similarities[0] == 0.9
            assert result.similarities[1] == 0.8
            assert result.ids[0] == "test-id-1"
            assert result.ids[1] == "test-id-2"


def test_query_unsupported_mode() -> None:
    """Test query method with unsupported mode."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
        with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
            )

            query = VectorStoreQuery(
                query_embedding=[1.0, 2.0, 3.0],
                mode=VectorStoreQueryMode.TEXT_SEARCH  # Unsupported mode
            )

            with pytest.raises(NotImplementedError, match="Query mode VectorStoreQueryMode.TEXT_SEARCH not available"):
                store.query(query)


def test_close() -> None:
    """Test close method."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_create_connection_pool"):
        with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
            )

            # Set initialized state
            store._is_initialized = True

            # Close the store
            store.close()

            # Verify it's marked as not initialized
            assert store._is_initialized is False


def test_from_params() -> None:
    """Test from_params class method."""
    with patch.object(AlibabaCloudMySQLVectorStore, "__init__", return_value=None) as mock_init:
        AlibabaCloudMySQLVectorStore.from_params(
            host="localhost",
            port=3306,
            user="test_user",
            password="test_password",
            database="test_db",
            table_name="custom_table",
            embed_dim=512,
            default_m=10,
            distance_method="EUCLIDEAN",
            perform_setup=False,
            charset="latin1",
            max_connection=20,
        )

        # Verify the parameters were passed correctly
        mock_init.assert_called_once_with(
            host="localhost",
            port=3306,
            user="test_user",
            password="test_password",
            database="test_db",
            table_name="custom_table",
            embed_dim=512,
            default_m=10,
            distance_method="EUCLIDEAN",
            perform_setup=False,
            charset="latin1",
            max_connection=20,
        )