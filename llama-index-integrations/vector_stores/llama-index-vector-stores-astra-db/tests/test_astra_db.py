import os
import pytest
from typing import Iterable

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
