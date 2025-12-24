"""Unit tests for AlibabaCloudMySQLVectorStore to improve test coverage."""

import json
from unittest.mock import Mock, patch
import pytest

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
    with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
        store = AlibabaCloudMySQLVectorStore(
            table_name="test_table",
            host="localhost",
            port=3306,
            user="test_user",
            password="test_password",
            database="test_db",
            perform_setup=False,  # Don't perform setup to avoid DB calls
        )

        # Test when initialized
        store._is_initialized = True
        store._engine = Mock()  # Mock the engine
        assert store.client is not None

        # Test when not initialized
        store._is_initialized = False
        assert store.client is None


def test_create_engine() -> None:
    """Test _create_engine method."""
    # Don't mock _connect, let it run to create engines
    store = AlibabaCloudMySQLVectorStore(
        table_name="test_table",
        host="localhost",
        port=3306,
        user="test_user",
        password="test_password",
        database="test_db",
        embed_dim=1536,
        default_m=6,
        distance_method="COSINE",
        perform_setup=False,  # Don't perform setup to avoid DB calls
    )

    # Verify the engines were created (they would be set in _connect)
    # The _connect method is called in __init__ via _initialize
    assert store._engine is not None
    assert store._async_engine is not None


def test_get_connection_context_manager() -> None:
    """Test session context manager."""
    mock_conn = Mock()
    mock_execute_result = Mock()

    with patch.object(AlibabaCloudMySQLVectorStore, "_connect") as mock_connect:
        store = AlibabaCloudMySQLVectorStore(
            table_name="test_table",
            host="localhost",
            port=3306,
            user="test_user",
            password="test_password",
            database="test_db",
            perform_setup=False,  # Don't perform setup to avoid DB calls
        )

        # Mock the session maker to return our mock connection
        mock_session = Mock()
        mock_session.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_session.return_value.__exit__ = Mock(return_value=None)
        store._session = mock_session

        # Test the context manager
        with store._session() as conn:
            assert conn == mock_conn

        # Verify the session context manager was used
        mock_session.assert_called_once()


def test_check_vector_support_success() -> None:
    """Test _check_vector_support method with successful case."""
    mock_session = Mock()
    mock_session.execute.return_value.fetchone.side_effect = [
        [True],  # vector_support result
        ["rds_release_date", "20251031"],  # rds release date result
    ]

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            # Should not raise any exception
            store._check_vector_support()


def test_check_vector_support_no_vector_functions() -> None:
    """Test _check_vector_support method when vector functions are not available."""
    mock_session = Mock()
    mock_session.execute.return_value.fetchone.return_value = [False]

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            with pytest.raises(
                ValueError, match="RDS MySQL Vector functions are not available"
            ):
                store._check_vector_support()


def test_check_vector_support_old_release_date() -> None:
    """Test _check_vector_support method with old release date."""
    mock_session = Mock()
    mock_session.execute.return_value.fetchone.side_effect = [
        [True],  # vector_support result
        ["rds_release_date", "20251030"],  # old rds release date
    ]

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            with pytest.raises(
                ValueError, match="rds_release_date must be 20251031 or later"
            ):
                store._check_vector_support()


def test_check_vector_support_no_release_date_variable() -> None:
    """Test _check_vector_support method when rds_release_date variable is not available."""
    mock_session = Mock()
    mock_session.execute.return_value.fetchone.side_effect = [
        [True],  # vector_support result
        None,  # No rds_release_date variable
    ]

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            with pytest.raises(
                ValueError, match="Unable to retrieve rds_release_date variable"
            ):
                store._check_vector_support()


def test_check_vector_support_function_error() -> None:
    """Test _check_vector_support method when VEC_FromText function raises an error."""
    mock_session = Mock()
    mock_session.execute.side_effect = Exception(
        "FUNCTION test.VEC_FromText does not exist"
    )

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            with pytest.raises(
                ValueError, match="RDS MySQL Vector functions are not available"
            ):
                store._check_vector_support()


def test_initialize_invalid_distance_method() -> None:
    """Test initialization with invalid distance method (should be caught by Pydantic validation)."""
    with pytest.raises(
        Exception
    ):  # Pydantic should catch invalid distance_method during initialization
        AlibabaCloudMySQLVectorStore(
            table_name="test_table",
            host="localhost",
            port=3306,
            user="test_user",
            password="test_password",
            database="test_db",
            distance_method="INVALID",
        )


def test_initialize_success() -> None:
    """Test _initialize method success case."""
    with patch.object(
        AlibabaCloudMySQLVectorStore, "_check_vector_support"
    ) as mock_check:
        with patch.object(
            AlibabaCloudMySQLVectorStore, "_create_table_if_not_exists"
        ) as mock_create_table:
            with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                    perform_setup=True,
                )

                # Verify methods were called
                mock_check.assert_called_once()
                mock_create_table.assert_called_once()
                assert store._is_initialized is True


def test_initialize_without_setup() -> None:
    """Test _initialize method when perform_setup is False."""
    with patch.object(
        AlibabaCloudMySQLVectorStore, "_check_vector_support"
    ) as mock_check:
        with patch.object(
            AlibabaCloudMySQLVectorStore, "_create_table_if_not_exists"
        ) as mock_create_table:
            with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                    perform_setup=False,
                )

                # Verify methods were NOT called when perform_setup is False
                mock_check.assert_not_called()
                mock_create_table.assert_not_called()
                assert store._is_initialized is True


def test_node_to_table_row() -> None:
    """Test _node_to_table_row method."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
        store = AlibabaCloudMySQLVectorStore(
            table_name="test_table",
            host="localhost",
            port=3306,
            user="test_user",
            password="test_password",
            database="test_db",
            perform_setup=False,  # Don't perform setup to avoid DB calls
        )

        # Create a test node
        node = TextNode(
            text="test text",
            id_="test-id",
            metadata={"key": "value"},
            embedding=[1.0, 2.0, 3.0],
        )

        # Convert node to table row
        row = store._node_to_table_row(node)

        assert row["node_id"] == "test-id"
        assert row["text"] == "test text"
        assert row["embedding"] == [1.0, 2.0, 3.0]
        assert "key" in row["metadata"]


def test_to_mysql_operator() -> None:
    """Test _to_mysql_operator method."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
        store = AlibabaCloudMySQLVectorStore(
            table_name="test_table",
            host="localhost",
            port=3306,
            user="test_user",
            password="test_password",
            database="test_db",
            perform_setup=False,  # Don't perform setup to avoid DB calls
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
        with patch(
            "llama_index.vector_stores.alibabacloud_mysql.base._logger"
        ) as mock_logger:
            assert store._to_mysql_operator("UNSUPPORTED") == "="
            mock_logger.warning.assert_called_once()


def test_build_filter_clause() -> None:
    """Test _build_filter_clause method."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
        store = AlibabaCloudMySQLVectorStore(
            table_name="test_table",
            host="localhost",
            port=3306,
            user="test_user",
            password="test_password",
            database="test_db",
            perform_setup=False,  # Don't perform setup to avoid DB calls
        )

        # Test simple equality filter
        filter_eq = MetadataFilter(
            key="category", value="test", operator=FilterOperator.EQ
        )
        global_param_counter = [0]
        clause, values = store._build_filter_clause(filter_eq, global_param_counter)
        assert "JSON_VALUE(metadata, '$.category') =" in clause
        # The values should now be a dictionary for SQLAlchemy named parameters
        assert isinstance(values, dict)

        # Test IN filter
        filter_in = MetadataFilter(
            key="category", value=["test1", "test2"], operator=FilterOperator.IN
        )
        global_param_counter = [0]
        clause, values = store._build_filter_clause(filter_in, global_param_counter)
        assert "JSON_VALUE(metadata, '$.category') IN" in clause
        # The values should now be a dictionary for SQLAlchemy named parameters
        assert isinstance(values, dict)


def test_filters_to_where_clause() -> None:
    """Test _filters_to_where_clause method."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
        store = AlibabaCloudMySQLVectorStore(
            table_name="test_table",
            host="localhost",
            port=3306,
            user="test_user",
            password="test_password",
            database="test_db",
            perform_setup=False,  # Don't perform setup to avoid DB calls
        )

        # Test simple filters
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="test", operator=FilterOperator.EQ
                ),
                MetadataFilter(key="priority", value=1, operator=FilterOperator.GT),
            ],
            condition="and",
        )

        global_param_counter = [0]
        clause, values = store._filters_to_where_clause(filters, global_param_counter)
        assert "JSON_VALUE(metadata, '$.category') =" in clause
        assert "JSON_VALUE(metadata, '$.priority') >" in clause
        assert "AND" in clause
        # Values should be a dictionary for SQLAlchemy named parameters
        assert isinstance(values, dict)

        # Test OR condition with multiple filters
        filters_or = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="test", operator=FilterOperator.EQ
                ),
                MetadataFilter(
                    key="type", value="document", operator=FilterOperator.EQ
                ),
            ],
            condition="or",
        )

        global_param_counter = [0]
        clause, values = store._filters_to_where_clause(
            filters_or, global_param_counter
        )
        assert "OR" in clause


def test_db_rows_to_query_result() -> None:
    """Test _db_rows_to_query_result method."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
        store = AlibabaCloudMySQLVectorStore(
            table_name="test_table",
            host="localhost",
            port=3306,
            user="test_user",
            password="test_password",
            database="test_db",
            perform_setup=False,  # Don't perform setup to avoid DB calls
        )

        # Create test rows
        from llama_index.vector_stores.alibabacloud_mysql.base import DBEmbeddingRow

        test_rows = [
            DBEmbeddingRow(
                node_id="test-id-1",
                text="test text 1",
                metadata={
                    "_node_content": '{"id_": "test-id-1", "text": "test text 1"}'
                },
                similarity=0.9,
            ),
            DBEmbeddingRow(
                node_id="test-id-2",
                text="test text 2",
                metadata={
                    "_node_content": '{"id_": "test-id-2", "text": "test text 2"}'
                },
                similarity=0.8,
            ),
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
    with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
        store = AlibabaCloudMySQLVectorStore(
            table_name="test_table",
            host="localhost",
            port=3306,
            user="test_user",
            password="test_password",
            database="test_db",
            perform_setup=False,  # Don't perform setup to avoid DB calls
        )

        query = VectorStoreQuery(
            query_embedding=[1.0, 2.0, 3.0],
            mode=VectorStoreQueryMode.TEXT_SEARCH,  # Unsupported mode
        )

        with pytest.raises(
            NotImplementedError,
            match=f"Query mode {VectorStoreQueryMode.TEXT_SEARCH} not available",
        ):
            store.query(query)


def test_close() -> None:
    """Test close method."""
    with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
        store = AlibabaCloudMySQLVectorStore(
            table_name="test_table",
            host="localhost",
            port=3306,
            user="test_user",
            password="test_password",
            database="test_db",
            perform_setup=False,  # Don't perform setup to avoid DB calls
        )

        # Set initialized state
        store._is_initialized = True

        # Close the store
        store.close()

        # Verify it's marked as not initialized
        assert store._is_initialized is False


def test_from_params() -> None:
    """Test from_params class method."""
    with patch.object(
        AlibabaCloudMySQLVectorStore, "__init__", return_value=None
    ) as mock_init:
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
            debug=False,
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
            debug=False,
        )


def test_get_nodes() -> None:
    """Test get_nodes method."""
    # Instead of mocking complex database results, we'll just verify the method can be called
    # without errors and that the proper query structure is used
    mock_session = Mock()

    # Mock the result to have proper iterable behavior
    mock_result = Mock()
    # Mock the result to behave like a SQLAlchemy result
    mock_result.__iter__ = Mock(return_value=iter([]))  # Empty iterator to avoid issues
    mock_session.execute.return_value = mock_result

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            # Test get_nodes with node_ids - just verify the method doesn't error
            nodes = store.get_nodes(node_ids=["test-id-1", "test-id-2"])
            # We can't verify exact count due to mock complexity, just that it returns a list
            assert isinstance(nodes, list)
            mock_session.execute.assert_called()
            # Check that the query uses SQLAlchemy named parameters
            args = mock_session.execute.call_args
            if args and args[0]:
                query = str(args[0][0])  # First argument of the call
                assert ":node_id_0" in query or ":node_id_1" in query

            # Test get_nodes without node_ids
            mock_session.reset_mock()
            nodes = store.get_nodes()
            assert isinstance(nodes, list)
            call_args = mock_session.execute.call_args
            if call_args and call_args[0]:
                query = str(call_args[0][0])  # First argument of the call
                # When no node_ids provided, query should not have WHERE clause
                assert "WHERE node_id IN" not in query


def test_add() -> None:
    """Test add method."""
    mock_session = Mock()

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            # Create test nodes
            nodes = [
                TextNode(
                    text="test text 1",
                    id_="test-id-1",
                    metadata={"key": "value1"},
                    embedding=[1.0, 2.0, 3.0],
                ),
                TextNode(
                    text="test text 2",
                    id_="test-id-2",
                    metadata={"key": "value2"},
                    embedding=[4.0, 5.0, 6.0],
                ),
            ]

            # Test adding nodes
            ids = store.add(nodes)
            assert len(ids) == 2
            assert "test-id-1" in ids
            assert "test-id-2" in ids
            assert mock_session.execute.call_count == 2


def test_query() -> None:
    """Test query method."""
    mock_session = Mock()
    mock_result = Mock()

    # Create mock rows that behave like database result rows
    class MockRow:
        def __init__(self, node_id, text, embedding, metadata, distance):
            self.node_id = node_id
            self.text = text
            self.embedding = embedding
            self.metadata = metadata
            self.distance = distance

        def __getitem__(self, index):
            if index == 0:
                return self.node_id  # node_id
            elif index == 1:
                return self.text  # text
            elif index == 2:
                return self.embedding  # embedding
            elif index == 3:
                return self.metadata  # metadata
            elif index == 4:
                return self.distance  # distance
            else:
                raise IndexError("Index out of range")

    row1 = MockRow(
        "test-id-1",
        "test text 1",
        "[1.0, 2.0, 3.0]",
        json.dumps({"_node_content": '{"id_": "test-id-1", "text": "test text 1"}'}),
        0.1,
    )
    row2 = MockRow(
        "test-id-2",
        "test text 2",
        "[4.0, 5.0, 6.0]",
        json.dumps({"_node_content": '{"id_": "test-id-2", "text": "test text 2"}'}),
        0.2,
    )
    mock_result.fetchall.return_value = [row1, row2]
    mock_session.execute.return_value = mock_result

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                distance_method="COSINE",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            # Test basic query
            query = VectorStoreQuery(
                query_embedding=[1.0, 2.0, 3.0], similarity_top_k=2
            )

            result = store.query(query)
            assert len(result.nodes) == 2
            assert len(result.similarities) == 2
            assert len(result.ids) == 2
            assert result.similarities[0] == 0.9  # 1 - 0.1
            assert result.similarities[1] == 0.8  # 1 - 0.2

            # Verify the query was executed
            mock_session.execute.assert_called_once()
            sql_query = str(mock_session.execute.call_args[0][0])
            assert "VEC_DISTANCE_COSINE" in sql_query


def test_query_with_filters() -> None:
    """Test query method with filters."""
    mock_session = Mock()
    mock_result = Mock()

    # Create mock rows that behave like database result rows
    class MockRow:
        def __init__(self, node_id, text, embedding, metadata, distance):
            self.node_id = node_id
            self.text = text
            self.embedding = embedding
            self.metadata = metadata
            self.distance = distance

        def __getitem__(self, index):
            if index == 0:
                return self.node_id  # node_id
            elif index == 1:
                return self.text  # text
            elif index == 2:
                return self.embedding  # embedding
            elif index == 3:
                return self.metadata  # metadata
            elif index == 4:
                return self.distance  # distance
            else:
                raise IndexError("Index out of range")

    row1 = MockRow(
        "test-id-1",
        "test text 1",
        "[1.0, 2.0, 3.0]",
        json.dumps({"_node_content": '{"id_": "test-id-1", "text": "test text 1"}'}),
        0.1,
    )
    mock_result.fetchall.return_value = [row1]
    mock_session.execute.return_value = mock_result

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                distance_method="EUCLIDEAN",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            # Test query with filters
            query = VectorStoreQuery(
                query_embedding=[1.0, 2.0, 3.0],
                similarity_top_k=1,
                filters=MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="category", value="test", operator=FilterOperator.EQ
                        )
                    ]
                ),
            )

            result = store.query(query)
            assert len(result.nodes) == 1

            # Verify the query was executed with filters
            mock_session.execute.assert_called_once()
            sql_query = str(mock_session.execute.call_args[0][0])
            assert "VEC_DISTANCE_EUCLIDEAN" in sql_query
            assert "JSON_VALUE(metadata, '$.category') =" in sql_query


def test_delete() -> None:
    """Test delete method."""
    mock_session = Mock()

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            # Test delete
            store.delete("test-ref-doc-id")

            # Verify the query was executed
            mock_session.execute.assert_called_once()
            sql_query = str(mock_session.execute.call_args[0][0])
            assert "JSON_EXTRACT(metadata, '$.ref_doc_id') =" in sql_query


def test_delete_nodes() -> None:
    """Test delete_nodes method."""
    mock_session = Mock()

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            # Test delete_nodes with node_ids
            store.delete_nodes(node_ids=["test-id-1", "test-id-2"])

            # Verify the query was executed
            mock_session.execute.assert_called_once()
            sql_query = str(mock_session.execute.call_args[0][0])
            assert "node_id IN" in sql_query


def test_count() -> None:
    """Test count method."""
    mock_session = Mock()
    mock_result = Mock()
    mock_result.fetchone.return_value = [5]
    mock_session.execute.return_value = mock_result

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            # Test count
            count = store.count()
            assert count == 5

            # Verify the query was executed
            mock_session.execute.assert_called_once()
            sql_query = str(mock_session.execute.call_args[0][0])
            assert "COUNT(*)" in sql_query


def test_drop() -> None:
    """Test drop method."""
    mock_session = Mock()

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            # Test drop
            store.drop()

            # Verify the query was executed
            mock_session.execute.assert_called_once()
            sql_query = str(mock_session.execute.call_args[0][0])
            assert "DROP TABLE IF EXISTS" in sql_query


def test_clear() -> None:
    """Test clear method."""
    mock_session = Mock()

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            # Test clear
            store.clear()

            # Verify the query was executed
            mock_session.execute.assert_called_once()
            sql_query = str(mock_session.execute.call_args[0][0])
            assert "DELETE FROM" in sql_query


def test_create_table_if_not_exists() -> None:
    """Test _create_table_if_not_exists method."""
    mock_session = Mock()

    with patch.object(AlibabaCloudMySQLVectorStore, "_session") as mock_session_maker:
        mock_session_maker.return_value.__enter__.return_value = mock_session
        with patch.object(AlibabaCloudMySQLVectorStore, "_connect"):
            store = AlibabaCloudMySQLVectorStore(
                table_name="test_table",
                host="localhost",
                port=3306,
                user="test_user",
                password="test_password",
                database="test_db",
                embed_dim=1536,
                default_m=6,
                distance_method="COSINE",
                perform_setup=False,  # Don't perform setup to avoid DB calls
            )

            # Test _create_table_if_not_exists
            store._create_table_if_not_exists()

            # Verify the query was executed
            mock_session.execute.assert_called_once()
            sql_query = str(mock_session.execute.call_args[0][0])
            assert "CREATE TABLE IF NOT EXISTS" in sql_query
            assert "VECTOR" in sql_query
