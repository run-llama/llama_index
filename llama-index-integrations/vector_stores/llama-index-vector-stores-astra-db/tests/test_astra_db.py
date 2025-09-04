import os
import pytest
from typing import Iterable
from unittest.mock import Mock, patch

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)
from llama_index.vector_stores.astra_db import AstraDBVectorStore


# env variables
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT", "")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")


@pytest.fixture(scope="module")
def astra_db_store() -> Iterable[AstraDBVectorStore]:
    store = AstraDBVectorStore(
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name="test_collection",
        keyspace=ASTRA_DB_KEYSPACE,
        embedding_dimension=2,
    )
    store._collection.delete_many({})
    yield store

    store._collection.drop()


# Unit tests with mocks (don't require credentials)
@patch("llama_index.vector_stores.astra_db.base.DataAPIClient")
def test_astra_db_initialization_mock(mock_client):
    """Test AstraDB initialization with mocks."""
    mock_database = Mock()
    mock_collection = Mock()
    mock_client.return_value.get_database.return_value = mock_database
    mock_database.create_collection.return_value = mock_collection

    store = AstraDBVectorStore(
        token="fake_token",
        api_endpoint="fake_endpoint",
        collection_name="test_collection",
        embedding_dimension=2,
    )

    assert store._embedding_dimension == 2
    mock_client.assert_called_once()
    mock_database.create_collection.assert_called_once()


@patch("llama_index.vector_stores.astra_db.base.DataAPIClient")
def test_from_params_mock(mock_client):
    """Test from_params class method with mocks."""
    mock_database = Mock()
    mock_collection = Mock()
    mock_client.return_value.get_database.return_value = mock_database
    mock_database.create_collection.return_value = mock_collection

    store = AstraDBVectorStore.from_params(
        token="fake_token",
        api_endpoint="fake_endpoint",
        collection_name="test_collection",
        embedding_dimension=2,
    )

    assert isinstance(store, AstraDBVectorStore)
    assert store._embedding_dimension == 2


@patch("llama_index.vector_stores.astra_db.base.DataAPIClient")
def test_add_mock(mock_client):
    """Test add method with mocks."""
    mock_database = Mock()
    mock_collection = Mock()
    mock_client.return_value.get_database.return_value = mock_database
    mock_database.create_collection.return_value = mock_collection

    # Mock successful insert
    mock_collection.insert_many.return_value = None

    store = AstraDBVectorStore(
        token="fake_token",
        api_endpoint="fake_endpoint",
        collection_name="test_collection",
        embedding_dimension=2,
    )

    test_node = TextNode(
        text="test content",
        id_="test_id",
        embedding=[0.1, 0.2],
    )

    result = store.add([test_node])

    assert result == ["test_id"]
    mock_collection.insert_many.assert_called_once()


@patch("llama_index.vector_stores.astra_db.base.DataAPIClient")
def test_delete_mock(mock_client):
    """Test delete method with mocks."""
    mock_database = Mock()
    mock_collection = Mock()
    mock_client.return_value.get_database.return_value = mock_database
    mock_database.create_collection.return_value = mock_collection

    store = AstraDBVectorStore(
        token="fake_token",
        api_endpoint="fake_endpoint",
        collection_name="test_collection",
        embedding_dimension=2,
    )

    store.delete("test_ref_doc_id")

    mock_collection.delete_one.assert_called_once_with({"_id": "test_ref_doc_id"})


@patch("llama_index.vector_stores.astra_db.base.DataAPIClient")
def test_get_nodes_mock(mock_client):
    """Test get_nodes method with mocks."""
    mock_database = Mock()
    mock_collection = Mock()
    mock_client.return_value.get_database.return_value = mock_database
    mock_database.create_collection.return_value = mock_collection

    # Mock find results
    mock_find_result = [
        {
            "_id": "node_1",
            "content": "test content",
            "metadata": {
                "_node_content": '{"test": "data"}',
                "category": "A",
            },
        }
    ]
    mock_collection.find.return_value = mock_find_result

    store = AstraDBVectorStore(
        token="fake_token",
        api_endpoint="fake_endpoint",
        collection_name="test_collection",
        embedding_dimension=2,
    )

    # Test get_nodes by node_ids
    nodes = store.get_nodes(node_ids=["node_1"])

    assert len(nodes) == 1
    mock_collection.find.assert_called_once()

    # Test error cases
    with pytest.raises(ValueError, match="Cannot specify both node_ids and filters"):
        filters = MetadataFilters(filters=[MetadataFilter(key="test", value="val")])
        store.get_nodes(node_ids=["node_1"], filters=filters)

    with pytest.raises(ValueError, match="Must specify either node_ids or filters"):
        store.get_nodes()


@patch("llama_index.vector_stores.astra_db.base.DataAPIClient")
def test_delete_nodes_mock(mock_client):
    """Test delete_nodes method with mocks."""
    mock_database = Mock()
    mock_collection = Mock()
    mock_client.return_value.get_database.return_value = mock_database
    mock_database.create_collection.return_value = mock_collection

    # Mock delete results
    mock_delete_result = Mock()
    mock_delete_result.deleted_count = 1
    mock_collection.delete_one.return_value = mock_delete_result
    mock_collection.delete_many.return_value = mock_delete_result

    store = AstraDBVectorStore(
        token="fake_token",
        api_endpoint="fake_endpoint",
        collection_name="test_collection",
        embedding_dimension=2,
    )

    # Test delete by single node_id
    store.delete_nodes(node_ids=["node_1"])
    mock_collection.delete_one.assert_called_once_with({"_id": "node_1"})

    # Test delete by multiple node_ids
    store.delete_nodes(node_ids=["node_1", "node_2"])
    mock_collection.delete_many.assert_called_with(
        {"_id": {"$in": ["node_1", "node_2"]}}
    )

    # Test delete by filters
    filters = MetadataFilters(filters=[MetadataFilter(key="category", value="A")])
    store.delete_nodes(filters=filters)

    # Test error cases
    with pytest.raises(ValueError, match="Cannot specify both node_ids and filters"):
        store.delete_nodes(node_ids=["node_1"], filters=filters)

    with pytest.raises(ValueError, match="Must specify either node_ids or filters"):
        store.delete_nodes()


@patch("llama_index.vector_stores.astra_db.base.DataAPIClient")
def test_query_mock(mock_client):
    """Test query method with mocks."""
    mock_database = Mock()
    mock_collection = Mock()
    mock_client.return_value.get_database.return_value = mock_database
    mock_database.create_collection.return_value = mock_collection

    # Mock query results
    mock_query_result = [
        {
            "_id": "node_1",
            "content": "test content",
            "metadata": {
                "_node_content": '{"test": "data"}',
            },
            "$similarity": 0.9,
        }
    ]
    mock_collection.find.return_value = mock_query_result

    store = AstraDBVectorStore(
        token="fake_token",
        api_endpoint="fake_endpoint",
        collection_name="test_collection",
        embedding_dimension=2,
    )

    query = VectorStoreQuery(
        query_embedding=[0.1, 0.2],
        similarity_top_k=1,
    )

    result = store.query(query)

    assert len(result.nodes) == 1
    assert len(result.similarities) == 1
    assert result.similarities[0] == 0.9
    mock_collection.find.assert_called_once()


@patch("llama_index.vector_stores.astra_db.base.DataAPIClient")
def test_query_filters_to_dict_mock(mock_client):
    """Test _query_filters_to_dict method with mocks."""
    mock_database = Mock()
    mock_collection = Mock()
    mock_client.return_value.get_database.return_value = mock_database
    mock_database.create_collection.return_value = mock_collection

    store = AstraDBVectorStore(
        token="fake_token",
        api_endpoint="fake_endpoint",
        collection_name="test_collection",
        embedding_dimension=2,
    )

    # Test basic filter conversion
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="A", operator=FilterOperator.EQ),
            MetadataFilter(key="score", value=85),
        ]
    )

    result = store._query_filters_to_dict(filters)
    expected = {
        "metadata.category": "A",
        "metadata.score": 85,
    }
    assert result == expected


@patch("llama_index.vector_stores.astra_db.base.DataAPIClient")
def test_add_with_exception_mock(mock_client):
    """Test add method with InsertManyException and replace logic."""
    from astrapy.exceptions import InsertManyException
    from astrapy.results import InsertManyResult, UpdateResult

    mock_database = Mock()
    mock_collection = Mock()
    mock_client.return_value.get_database.return_value = mock_database
    mock_database.create_collection.return_value = mock_collection

    # Mock InsertManyException with partial result
    partial_result = Mock(spec=InsertManyResult)
    partial_result.inserted_ids = ["test_id_2"]  # Only second doc inserted
    insert_exception = InsertManyException(
        text="Some docs failed to insert",
        detailed_error_descriptors=[
            "Some docs failed to insert",
            "Some docs failed to insert",
        ],
        partial_result=partial_result,
        error_descriptors=[
            "Some docs failed to insert",
            "Some docs failed to insert",
        ],
    )

    # Mock replace_one result
    update_result = Mock(spec=UpdateResult)
    update_result.update_info = {"n": 1}

    mock_collection.insert_many.side_effect = insert_exception
    mock_collection.replace_one.return_value = update_result

    store = AstraDBVectorStore(
        token="fake_token",
        api_endpoint="fake_endpoint",
        collection_name="test_collection",
        embedding_dimension=2,
    )

    test_nodes = [
        TextNode(text="test content 1", id_="test_id_1", embedding=[0.1, 0.2]),
        TextNode(text="test content 2", id_="test_id_2", embedding=[0.3, 0.4]),
    ]

    result = store.add(test_nodes)

    assert result == ["test_id_1", "test_id_2"]
    mock_collection.insert_many.assert_called_once()
    mock_collection.replace_one.assert_called_once()


@patch("llama_index.vector_stores.astra_db.base.DataAPIClient")
def test_delete_with_kwargs_warning(mock_client):
    """Test delete method with unsupported kwargs triggers warning."""
    mock_database = Mock()
    mock_collection = Mock()
    mock_client.return_value.get_database.return_value = mock_database
    mock_database.create_collection.return_value = mock_collection

    store = AstraDBVectorStore(
        token="fake_token",
        api_endpoint="fake_endpoint",
        collection_name="test_collection",
        embedding_dimension=2,
    )

    with pytest.warns(UserWarning, match="unsupported named argument"):
        store.delete("test_ref_doc_id", unsupported_kwarg="value")


@patch("llama_index.vector_stores.astra_db.base.DataAPIClient")
def test_delete_nodes_with_kwargs_warning(mock_client):
    """Test delete_nodes method with unsupported kwargs triggers warning."""
    mock_database = Mock()
    mock_collection = Mock()
    mock_client.return_value.get_database.return_value = mock_database
    mock_database.create_collection.return_value = mock_collection

    mock_delete_result = Mock()
    mock_delete_result.deleted_count = 1
    mock_collection.delete_one.return_value = mock_delete_result

    store = AstraDBVectorStore(
        token="fake_token",
        api_endpoint="fake_endpoint",
        collection_name="test_collection",
        embedding_dimension=2,
    )

    with pytest.warns(UserWarning, match="unsupported named argument"):
        store.delete_nodes(node_ids=["node_1"], unsupported_kwarg="value")


@patch("llama_index.vector_stores.astra_db.base.DataAPIClient")
def test_client_property(mock_client):
    """Test client property returns the collection."""
    mock_database = Mock()
    mock_collection = Mock()
    mock_client.return_value.get_database.return_value = mock_database
    mock_database.create_collection.return_value = mock_collection

    store = AstraDBVectorStore(
        token="fake_token",
        api_endpoint="fake_endpoint",
        collection_name="test_collection",
        embedding_dimension=2,
    )

    assert store.client == mock_collection


@patch("llama_index.vector_stores.astra_db.base.DataAPIClient")
def test_initialization_with_ttl_warning(mock_client):
    """Test initialization with ttl_seconds triggers warning."""
    mock_database = Mock()
    mock_collection = Mock()
    mock_client.return_value.get_database.return_value = mock_database
    mock_database.create_collection.return_value = mock_collection

    with pytest.warns(UserWarning, match="ttl_seconds.*not supported"):
        store = AstraDBVectorStore(
            token="fake_token",
            api_endpoint="fake_endpoint",
            collection_name="test_collection",
            embedding_dimension=2,
            ttl_seconds=3600,
        )


@patch("llama_index.vector_stores.astra_db.base.DataAPIClient")
def test_query_unsupported_mode(mock_client):
    """Test query with unsupported mode raises NotImplementedError."""
    mock_database = Mock()
    mock_collection = Mock()
    mock_client.return_value.get_database.return_value = mock_database
    mock_database.create_collection.return_value = mock_collection

    store = AstraDBVectorStore(
        token="fake_token",
        api_endpoint="fake_endpoint",
        collection_name="test_collection",
        embedding_dimension=2,
    )

    from llama_index.core.vector_stores.types import VectorStoreQueryMode

    query = VectorStoreQuery(
        query_embedding=[0.1, 0.2],
        similarity_top_k=1,
        mode=VectorStoreQueryMode.SPARSE,  # Unsupported mode
    )

    with pytest.raises(NotImplementedError, match="Query mode.*not available"):
        store.query(query)


# Integration tests (require credentials)
@pytest.mark.skipif(
    ASTRA_DB_APPLICATION_TOKEN == "" or ASTRA_DB_API_ENDPOINT == "",
    reason="missing Astra DB credentials",
)
def test_astra_db_create_and_crud(astra_db_store: AstraDBVectorStore) -> None:
    """Test basic creation and insertion/deletion of a node."""
    astra_db_store.add(
        [
            TextNode(
                text="test node text",
                id_="test node id",
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test doc id")
                },
                embedding=[0.5, 0.5],
            )
        ]
    )

    astra_db_store.delete("test node id")


@pytest.mark.skipif(
    ASTRA_DB_APPLICATION_TOKEN == "" or ASTRA_DB_API_ENDPOINT == "",
    reason="missing Astra DB credentials",
)
def test_astra_db_queries(astra_db_store: AstraDBVectorStore) -> None:
    """Test basic querying."""
    query = VectorStoreQuery(query_embedding=[1, 1], similarity_top_k=3)

    astra_db_store.query(
        query,
    )


@pytest.mark.skipif(
    ASTRA_DB_APPLICATION_TOKEN == "" or ASTRA_DB_API_ENDPOINT == "",
    reason="missing Astra DB credentials",
)
def test_astra_db_insertions(astra_db_store: AstraDBVectorStore) -> None:
    """Test massive insertion with overwrites."""
    all_ids = list(range(150))
    nodes0 = [
        TextNode(
            text=f"OLD_node {idx}",
            id_=f"n_{idx}",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_0")},
            embedding=[0.5, 0.5],
        )
        for idx in all_ids[60:80] + all_ids[130:140]
    ]
    nodes = [
        TextNode(
            text=f"node {idx}",
            id_=f"n_{idx}",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_0")},
            embedding=[0.5, 0.5],
        )
        for idx in all_ids
    ]

    astra_db_store.add(nodes0)
    found_contents0 = [doc["content"] for doc in astra_db_store._collection.find({})]
    assert all(f_content[:4] == "OLD_" for f_content in found_contents0)
    assert len(found_contents0) == len(nodes0)

    astra_db_store.add(nodes)
    found_contents = [doc["content"] for doc in astra_db_store._collection.find({})]
    assert all(f_content[:5] == "node " for f_content in found_contents)
    assert len(found_contents) == len(nodes)


def test_class_name() -> None:
    """Test class_name class method."""
    assert AstraDBVectorStore.class_name() == "AstraDBVectorStore"


@pytest.mark.skipif(
    ASTRA_DB_APPLICATION_TOKEN == "" or ASTRA_DB_API_ENDPOINT == "",
    reason="missing Astra DB credentials",
)
def test_from_params() -> None:
    """Test from_params class method."""
    store = AstraDBVectorStore.from_params(
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name="test_from_params",
        keyspace=ASTRA_DB_KEYSPACE,
        embedding_dimension=2,
    )

    assert isinstance(store, AstraDBVectorStore)
    assert store._embedding_dimension == 2

    # Clean up
    store._collection.drop()


@pytest.mark.skipif(
    ASTRA_DB_APPLICATION_TOKEN == "" or ASTRA_DB_API_ENDPOINT == "",
    reason="missing Astra DB credentials",
)
def test_get_nodes(astra_db_store: AstraDBVectorStore) -> None:
    """Test get_nodes method."""
    # Clear collection and add test nodes
    astra_db_store._collection.delete_many({})

    test_nodes = [
        TextNode(
            text="test node 1",
            id_="node_1",
            metadata={"category": "A", "score": 85},
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_1")},
            embedding=[0.1, 0.1],
        ),
        TextNode(
            text="test node 2",
            id_="node_2",
            metadata={"category": "B", "score": 90},
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_2")},
            embedding=[0.2, 0.2],
        ),
        TextNode(
            text="test node 3",
            id_="node_3",
            metadata={"category": "A", "score": 75},
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_3")},
            embedding=[0.3, 0.3],
        ),
    ]

    astra_db_store.add(test_nodes)

    # Test get_nodes by node_ids
    retrieved_nodes = astra_db_store.get_nodes(node_ids=["node_1", "node_3"])
    assert len(retrieved_nodes) == 2
    retrieved_ids = {node.node_id for node in retrieved_nodes}
    assert retrieved_ids == {"node_1", "node_3"}

    # Test get_nodes by single node_id
    retrieved_nodes = astra_db_store.get_nodes(node_ids=["node_2"])
    assert len(retrieved_nodes) == 1
    assert retrieved_nodes[0].node_id == "node_2"
    assert retrieved_nodes[0].text == "test node 2"

    # Test get_nodes by filters
    filters = MetadataFilters(
        filters=[MetadataFilter(key="category", value="A", operator=FilterOperator.EQ)]
    )
    retrieved_nodes = astra_db_store.get_nodes(filters=filters)
    assert len(retrieved_nodes) == 2
    retrieved_ids = {node.node_id for node in retrieved_nodes}
    assert retrieved_ids == {"node_1", "node_3"}

    # Test get_nodes with limit
    retrieved_nodes = astra_db_store.get_nodes(filters=filters, limit=1)
    assert len(retrieved_nodes) == 1

    # Test error cases
    with pytest.raises(ValueError, match="Cannot specify both node_ids and filters"):
        astra_db_store.get_nodes(node_ids=["node_1"], filters=filters)

    with pytest.raises(ValueError, match="Must specify either node_ids or filters"):
        astra_db_store.get_nodes()


@pytest.mark.skipif(
    ASTRA_DB_APPLICATION_TOKEN == "" or ASTRA_DB_API_ENDPOINT == "",
    reason="missing Astra DB credentials",
)
def test_delete_nodes(astra_db_store: AstraDBVectorStore) -> None:
    """Test delete_nodes method."""
    # Clear collection and add test nodes
    astra_db_store._collection.delete_many({})

    test_nodes = [
        TextNode(
            text="delete test node 1",
            id_="del_node_1",
            metadata={"category": "delete_test", "score": 85},
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="del_doc_1")
            },
            embedding=[0.1, 0.1],
        ),
        TextNode(
            text="delete test node 2",
            id_="del_node_2",
            metadata={"category": "delete_test", "score": 90},
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="del_doc_2")
            },
            embedding=[0.2, 0.2],
        ),
        TextNode(
            text="keep test node",
            id_="keep_node",
            metadata={"category": "keep_test", "score": 95},
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="keep_doc")
            },
            embedding=[0.3, 0.3],
        ),
    ]

    astra_db_store.add(test_nodes)

    # Verify all nodes are present
    all_docs = list(astra_db_store._collection.find({}))
    assert len(all_docs) == 3

    # Test delete_nodes by node_ids (multiple)
    astra_db_store.delete_nodes(node_ids=["del_node_1", "del_node_2"])

    # Verify deletion
    remaining_docs = list(astra_db_store._collection.find({}))
    assert len(remaining_docs) == 1
    assert remaining_docs[0]["_id"] == "keep_node"

    # Add nodes back for filter test
    astra_db_store.add(test_nodes[:2])  # Add back the deleted nodes

    # Test delete_nodes by single node_id
    astra_db_store.delete_nodes(node_ids=["del_node_1"])
    remaining_docs = list(astra_db_store._collection.find({}))
    remaining_ids = {doc["_id"] for doc in remaining_docs}
    assert "del_node_1" not in remaining_ids
    assert "del_node_2" in remaining_ids
    assert "keep_node" in remaining_ids

    # Test delete_nodes by filters
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value="delete_test", operator=FilterOperator.EQ
            )
        ]
    )
    astra_db_store.delete_nodes(filters=filters)

    # Verify only keep_node remains
    remaining_docs = list(astra_db_store._collection.find({}))
    assert len(remaining_docs) == 1
    assert remaining_docs[0]["_id"] == "keep_node"

    # Test error cases
    with pytest.raises(ValueError, match="Cannot specify both node_ids and filters"):
        astra_db_store.delete_nodes(node_ids=["node_1"], filters=filters)

    with pytest.raises(ValueError, match="Must specify either node_ids or filters"):
        astra_db_store.delete_nodes()
