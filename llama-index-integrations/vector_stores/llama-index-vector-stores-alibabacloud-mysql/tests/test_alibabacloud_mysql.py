"""Unit tests for AlibabaCloudMySQLVectorStore to improve test coverage."""

import json
from unittest.mock import Mock, patch, MagicMock

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
    with patch("sqlalchemy.create_engine") as mock_engine:
        mock_engine.return_value.dispose = Mock()
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


def test_close() -> None:
    """Test close method."""
    with patch("sqlalchemy.create_engine") as mock_engine:
        mock_engine.return_value.dispose = Mock()
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
            mock_engine.return_value.dispose.assert_called_once()


def test_from_params() -> None:
    """Test from_params class method."""
    with patch("sqlalchemy.create_engine") as mock_engine:
        mock_engine.return_value.dispose = Mock()
        with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
            store = AlibabaCloudMySQLVectorStore.from_params(
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
            assert store.host == "localhost"
            assert store.port == 3306
            assert store.user == "test_user"
            assert store.database == "test_db"
            assert store.table_name == "custom_table"
            assert store.embed_dim == 512
            assert store.default_m == 10
            assert store.distance_method == "EUCLIDEAN"
            assert store.perform_setup is False
            assert store.charset == "latin1"
            assert store.max_connection == 20


def test_node_to_table_row() -> None:
    """Test _node_to_table_row method."""
    with patch("sqlalchemy.create_engine") as mock_engine:
        mock_engine.return_value.dispose = Mock()
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
    with patch("sqlalchemy.create_engine") as mock_engine:
        mock_engine.return_value.dispose = Mock()
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
            with patch(
                "llama_index.vector_stores.alibabacloud_mysql.base._logger"
            ) as mock_logger:
                assert store._to_mysql_operator("UNSUPPORTED") == "="
                mock_logger.warning.assert_called_once()


def test_build_filter_clause() -> None:
    """Test _build_filter_clause method."""
    with patch("sqlalchemy.create_engine") as mock_engine:
        mock_engine.return_value.dispose = Mock()
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
            filter_eq = MetadataFilter(
                key="category", value="test", operator=FilterOperator.EQ
            )
            clause = store._build_filter_clause(filter_eq)
            assert "JSON_VALUE(metadata, '$.category') =" in clause

            # Test IN filter
            filter_in = MetadataFilter(
                key="category", value=["test1", "test2"], operator=FilterOperator.IN
            )
            clause = store._build_filter_clause(filter_in)
            assert "JSON_VALUE(metadata, '$.category') IN" in clause


def test_filters_to_where_clause() -> None:
    """Test _filters_to_where_clause method."""
    with patch("sqlalchemy.create_engine") as mock_engine:
        mock_engine.return_value.dispose = Mock()
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
                    MetadataFilter(
                        key="category", value="test", operator=FilterOperator.EQ
                    ),
                    MetadataFilter(key="priority", value=1, operator=FilterOperator.GT),
                ],
                condition="and",
            )

            clause, values = store._filters_to_where_clause(filters)
            assert "JSON_VALUE(metadata, '$.category') =" in clause
            assert "JSON_VALUE(metadata, '$.priority') >" in clause
            assert "AND" in clause
            assert values == ["test", 1]

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

            clause, values = store._filters_to_where_clause(filters_or)
            assert "OR" in clause


def test_db_rows_to_query_result() -> None:
    """Test _db_rows_to_query_result method."""
    with patch("sqlalchemy.create_engine") as mock_engine:
        mock_engine.return_value.dispose = Mock()
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
    with patch("sqlalchemy.create_engine") as mock_engine:
        mock_engine.return_value.dispose = Mock()
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
                mode=VectorStoreQueryMode.TEXT_SEARCH,  # Unsupported mode
            )

            with pytest.raises(
                NotImplementedError,
                match="Query mode VectorStoreQueryMode.TEXT_SEARCH not available",
            ):
                store.query(query)


def test_get_nodes() -> None:
    """Test get_nodes method."""
    mock_session = MagicMock()
    mock_execute = MagicMock()
    mock_execute.fetchall.return_value = [
        Mock(
            text="test text 1",
            metadata=json.dumps(
                {"_node_content": '{"id_": "test-id-1", "text": "test text 1"}'}
            ),
        ),
        Mock(
            text="test text 2",
            metadata=json.dumps(
                {"_node_content": '{"id_": "test-id-2", "text": "test text 2"}'}
            ),
        ),
    ]
    mock_session.__enter__.return_value.execute.return_value = mock_execute

    with patch("sqlalchemy.create_engine") as mock_engine:
        with patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker:
            mock_engine.return_value.dispose = Mock()
            mock_sessionmaker.return_value.return_value = mock_session
            with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                )

                # Test get_nodes with node_ids
                nodes = store.get_nodes(node_ids=["test-id-1", "test-id-2"])
                assert len(nodes) == 2
                assert mock_session.__enter__.return_value.execute.called


def test_add() -> None:
    """Test add method."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value.execute = Mock()
    mock_session.__enter__.return_value.commit = Mock()

    with patch("sqlalchemy.create_engine") as mock_engine:
        with patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker:
            mock_engine.return_value.dispose = Mock()
            mock_sessionmaker.return_value.return_value = mock_session
            with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
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
                assert mock_session.__enter__.return_value.execute.call_count == 2


def test_query() -> None:
    """Test query method."""
    mock_session = MagicMock()
    mock_execute = MagicMock()
    mock_execute.fetchall.return_value = [
        Mock(
            node_id="test-id-1",
            text="test text 1",
            metadata=json.dumps(
                {"_node_content": '{"id_": "test-id-1", "text": "test text 1"}'}
            ),
            distance=0.1,
        ),
        Mock(
            node_id="test-id-2",
            text="test text 2",
            metadata=json.dumps(
                {"_node_content": '{"id_": "test-id-2", "text": "test text 2"}'}
            ),
            distance=0.2,
        ),
    ]
    mock_session.__enter__.return_value.execute.return_value = mock_execute

    with patch("sqlalchemy.create_engine") as mock_engine:
        with patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker:
            mock_engine.return_value.dispose = Mock()
            mock_sessionmaker.return_value.return_value = mock_session
            with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                    distance_method="COSINE",
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


def test_delete() -> None:
    """Test delete method."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value.execute = Mock()
    mock_session.__enter__.return_value.commit = Mock()

    with patch("sqlalchemy.create_engine") as mock_engine:
        with patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker:
            mock_engine.return_value.dispose = Mock()
            mock_sessionmaker.return_value.return_value = mock_session
            with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                )

                # Test delete
                store.delete("test-ref-doc-id")

                # Verify the query was executed
                assert mock_session.__enter__.return_value.execute.called


def test_count() -> None:
    """Test count method."""
    mock_session = MagicMock()
    mock_execute = MagicMock()
    mock_execute.fetchone.return_value = (5,)
    mock_session.__enter__.return_value.execute.return_value = mock_execute

    with patch("sqlalchemy.create_engine") as mock_engine:
        with patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker:
            mock_engine.return_value.dispose = Mock()
            mock_sessionmaker.return_value.return_value = mock_session
            with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                )

                # Test count
                count = store.count()
                assert count == 5


def test_drop() -> None:
    """Test drop method."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value.execute = Mock()
    mock_session.__enter__.return_value.commit = Mock()

    with patch("sqlalchemy.create_engine") as mock_engine:
        with patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker:
            mock_engine.return_value.dispose = Mock()
            mock_sessionmaker.return_value.return_value = mock_session
            with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                )

                # Test drop
                store.drop()

                # Verify the query was executed
                assert mock_session.__enter__.return_value.execute.called


def test_clear() -> None:
    """Test clear method."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value.execute = Mock()
    mock_session.__enter__.return_value.commit = Mock()

    with patch("sqlalchemy.create_engine") as mock_engine:
        with patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker:
            mock_engine.return_value.dispose = Mock()
            mock_sessionmaker.return_value.return_value = mock_session
            with patch.object(AlibabaCloudMySQLVectorStore, "_initialize"):
                store = AlibabaCloudMySQLVectorStore(
                    table_name="test_table",
                    host="localhost",
                    port=3306,
                    user="test_user",
                    password="test_password",
                    database="test_db",
                )

                # Test clear
                store.clear()

                # Verify the query was executed
                assert mock_session.__enter__.return_value.execute.called
