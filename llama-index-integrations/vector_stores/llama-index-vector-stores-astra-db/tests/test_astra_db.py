import os
import pytest
from typing import Iterable

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
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
