"""Integration tests for llama-index-vector-stores-alibabacloud-mysql."""

from typing import Generator, List

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from llama_index.vector_stores.alibabacloud_mysql import AlibabaCloudMySQLVectorStore

TEST_NODES: List[TextNode] = [
    TextNode(
        text="lorem ipsum",
        id_="c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
        metadata={
            "author": "Stephen King",
            "theme": "Friendship",
            "pages": 1000,
        },
        embedding=[1.0, 0.0, 0.0],
    ),
    TextNode(
        text="dolor sit amet",
        id_="c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
        metadata={
            "director": "Francis Ford Coppola",
            "theme": "Mafia",
        },
        embedding=[0.0, 1.0, 0.0],
    ),
    TextNode(
        text="consectetur adipiscing elit",
        id_="c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
        metadata={
            "director": "Christopher Nolan",
        },
        embedding=[0.0, 0.0, 1.0],
    ),
]


vector_store = None
try:
    vector_store = AlibabaCloudMySQLVectorStore(
        collection_name="vector_store_test",
        host="127.0.0.1",
        port=3306,
        user="root",
        password="test",
        database="test",
        embed_dim=3,
        default_m=16,
        distance_method="COSINE"
    )

    # If you want to run the integration tests you need to do:
    # Have an Alibaba Cloud MySQL instance with vector search support

    # Check if we are able to connect to the Alibaba Cloud MySQL instance
    vector_store._check_vector_support()
    run_integration_tests = True
except Exception:
    run_integration_tests = False


@pytest.fixture(autouse=True)
def teardown(request: pytest.FixtureRequest) -> Generator:
    """Clear the store after a test completion."""
    yield

    if "noautousefixtures" in request.keywords:
        return

    if vector_store is not None:
        vector_store.clear()


@pytest.fixture(scope="session", autouse=True)
def close_db_connection(request: pytest.FixtureRequest) -> Generator:
    """Close the DB connections after the last test."""
    yield

    if "noautousefixtures" in request.keywords:
        return

    if vector_store is not None:
        vector_store.close()


@pytest.mark.skipif(
    not run_integration_tests,
    reason="Alibaba Cloud MySQL instance required for integration tests",
)
def test_query() -> None:
    vector_store.add(TEST_NODES)
    res = vector_store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
    )
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].get_content() == "lorem ipsum"


@pytest.mark.skipif(
    not run_integration_tests,
    reason="Alibaba Cloud MySQL instance required for integration tests",
)
def test_query_with_metadatafilters() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="director",
                value=["Francis Ford Coppola", "Christopher Nolan"],
                operator=FilterOperator.IN,
            ),
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="theme", value="Mafia", operator=FilterOperator.EQ
                    ),
                    MetadataFilter(key="pages", value=1000, operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )

    vector_store.add(TEST_NODES)
    res = vector_store.query(
        VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0], filters=filters, similarity_top_k=3
        )
    )

    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].get_content() == "dolor sit amet"


@pytest.mark.skipif(
    not run_integration_tests,
    reason="Alibaba Cloud MySQL instance required for integration tests",
)
def test_delete() -> None:
    vector_store.add(TEST_NODES)
    vector_store.delete("test-0")
    vector_store.delete("test-1")
    res = vector_store.get_nodes(
        node_ids=[
            "c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            "c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
        ]
    )
    assert len(res) == 1
    assert res[0].get_content() == "consectetur adipiscing elit"
    assert res[0].id_ == "c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d"


@pytest.mark.skipif(
    not run_integration_tests,
    reason="Alibaba Cloud MySQL instance required for integration tests",
)
def test_delete_nodes() -> None:
    vector_store.add(TEST_NODES)
    vector_store.delete_nodes(
        node_ids=[
            "c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
        ]
    )
    res = vector_store.get_nodes(
        node_ids=[
            "c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            "c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
        ]
    )
    assert len(res) == 1
    assert res[0].get_content() == "consectetur adipiscing elit"
    assert res[0].id_ == "c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d"


@pytest.mark.skipif(
    not run_integration_tests,
    reason="Alibaba Cloud MySQL instance required for integration tests",
)
def test_count() -> None:
    vector_store.add(TEST_NODES)
    assert vector_store.count() == 3


@pytest.mark.skipif(
    not run_integration_tests,
    reason="Alibaba Cloud MySQL instance required for integration tests",
)
def test_drop() -> None:
    vector_store.add(TEST_NODES)
    vector_store.drop()
    assert vector_store.count() == 0


@pytest.mark.skipif(
    not run_integration_tests,
    reason="Alibaba Cloud MySQL instance required for integration tests",
)
def test_clear() -> None:
    vector_store.add(TEST_NODES)
    vector_store.clear()
    res = vector_store.get_nodes(
        node_ids=[
            "c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            "c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
        ]
    )
    assert len(res) == 0