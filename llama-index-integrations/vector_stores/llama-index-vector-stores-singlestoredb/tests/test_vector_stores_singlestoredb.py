from unittest.mock import MagicMock, patch
import pytest
import json
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
)
from llama_index.vector_stores.singlestoredb import SingleStoreVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in SingleStoreVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@patch("singlestoredb.connect")
@patch.object(SingleStoreVectorStore, "_create_table")
def test_query_with_embedding_parameterized(mock_create_table, mock_s2_connect):
    """Test that query properly uses parameterized queries to prevent SQL injection."""
    mock_s2_connect.return_value = MagicMock()

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mock_metadata = {
        "node_id": "test-node-id",
        "node_type": "TextNode",
        "ref_doc_id": "test-doc-id",
        "metadata": {},
        "embedding": None,
    }
    mock_cursor.fetchall.return_value = [
        ("test content", json.dumps(mock_metadata), 0.95)
    ]

    with patch("sqlalchemy.pool.QueuePool") as mock_pool_class:
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        mock_pool.connect.return_value = mock_conn

        store = SingleStoreVectorStore(
            table_name="test_table",
            content_field="content",
            metadata_field="metadata",
            vector_field="vector",
        )

        query = VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=5)

        result = store.query(query)

        assert result is not None


@patch("singlestoredb.connect")
@patch.object(SingleStoreVectorStore, "_create_table")
def test_query_with_embedding_and_filter(mock_create_table, mock_s2_connect):
    """Test query with both embedding and filter using parameterized queries."""
    mock_s2_connect.return_value = MagicMock()

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mock_metadata = {
        "node_id": "filtered-node-id",
        "node_type": "TextNode",
        "ref_doc_id": "test-doc-id",
        "metadata": {"category": "test"},
        "embedding": None,
    }
    mock_cursor.fetchall.return_value = [
        ("filtered content", json.dumps(mock_metadata), 0.85)
    ]

    with patch("sqlalchemy.pool.QueuePool") as mock_pool_class:
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        mock_pool.connect.return_value = mock_conn

        store = SingleStoreVectorStore(
            table_name="test_table",
            content_field="content",
            metadata_field="metadata",
            vector_field="vector",
        )

        query = VectorStoreQuery(query_embedding=[0.4, 0.5, 0.6], similarity_top_k=10)

        filter_dict = {"category": "test"}
        result = store.query(query, filter=filter_dict)

        assert result is not None


@patch("singlestoredb.connect")
@patch.object(SingleStoreVectorStore, "_create_table")
def test_query_similarity_top_k_validation(mock_create_table, mock_s2_connect):
    """Test that invalid similarity_top_k values raise an error."""
    mock_s2_connect.return_value = MagicMock()

    with patch("sqlalchemy.pool.QueuePool") as mock_pool_class:
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        store = SingleStoreVectorStore(table_name="test_table")

        query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=-1)

        with pytest.raises(
            ValueError, match="similarity_top_k must be a positive integer"
        ):
            store.query(query)

        query.similarity_top_k = 0
        with pytest.raises(
            ValueError, match="similarity_top_k must be a positive integer"
        ):
            store.query(query)
