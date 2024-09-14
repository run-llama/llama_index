import asyncio
import logging
import pytest
import time
import uuid
from datetime import datetime
from typing import List, Generator, Set

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.opensearch import (
    OpensearchVectorClient,
    OpensearchVectorStore,
)
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

##
# Start Opensearch locally
# cd tests
# docker-compose up
#
# Run tests
# pytest test_opensearch_client.py

logging.basicConfig(level=logging.DEBUG)
evt_loop = asyncio.get_event_loop()

try:
    from opensearchpy import OpenSearch

    sync_os_client = OpenSearch("localhost:9200")
    sync_os_client.info()
    opensearch_not_available = False
except (ImportError, Exception):
    opensearch_not_available = True
finally:
    sync_os_client.close()

TEST_EMBED_DIM = 3


def _get_sample_vector(num: float) -> List[float]:
    """
    Get sample embedding vector of the form [num, 1, 1, ..., 1]
    where the length of the vector is TEST_EMBED_DIM.
    """
    return [num] + [1.0] * (TEST_EMBED_DIM - 1)


def _get_sample_vector_store_query(filters: MetadataFilters) -> VectorStoreQuery:
    return VectorStoreQuery(
        query_embedding=[0.1] * TEST_EMBED_DIM, similarity_top_k=100, filters=filters
    )


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
def test_connection() -> None:
    assert True


@pytest.fixture()
def index_name() -> str:
    """Return the index name."""
    return f"test_{uuid.uuid4().hex}"


@pytest.fixture()
def os_store(index_name: str) -> Generator[OpensearchVectorStore, None, None]:
    client = OpensearchVectorClient(
        endpoint="localhost:9200",
        index=index_name,
        dim=3,
    )

    yield OpensearchVectorStore(client)

    # teardown step
    # delete index
    client._os_client.indices.delete(index=index_name)
    # close client
    client._os_client.close()
    client._os_async_client.close()


@pytest.fixture(scope="session")
def node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
            metadata={
                "author": "Stephen King",
                "theme": "Friendship",
            },
            embedding=[1.0, 0.0, 0.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
            },
            embedding=[0.0, 1.0, 0.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
            metadata={
                "director": "Christopher Nolan",
            },
            embedding=[0.0, 0.0, 1.0],
        ),
        TextNode(
            text="I was taught that the way of progress was neither swift nor easy.",
            id_="0b31ae71-b797-4e88-8495-031371a7752e",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-3")},
            metadata={
                "author": "Marie Curie",
            },
            embedding=[0.0, 0.0, 0.9],
        ),
        TextNode(
            text=(
                "The important thing is not to stop questioning."
                + " Curiosity has its own reason for existing."
            ),
            id_="bd2e080b-159a-4030-acc3-d98afd2ba49b",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-4")},
            metadata={
                "author": "Albert Einstein",
            },
            embedding=[0.0, 0.0, 0.5],
        ),
        TextNode(
            text=(
                "I am no bird; and no net ensnares me;"
                + " I am a free human being with an independent will."
            ),
            id_="f658de3b-8cef-4d1c-8bed-9a263c907251",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-5")},
            metadata={
                "author": "Charlotte Bronte",
            },
            embedding=[0.0, 0.0, 0.3],
        ),
    ]


@pytest.fixture(scope="session")
def node_embeddings_2() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="aaa")},
            extra_info={"test_num": "1"},
            embedding=_get_sample_vector(1.0),
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="bbb")},
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="consectetur adipiscing elit",
            id_="ccc",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ccc")},
            extra_info={"test_key_list": ["test_value"]},
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="sed do eiusmod tempor",
            id_="ddd",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ccc")},
            extra_info={"test_key_2": "test_val_2"},
            embedding=_get_sample_vector(0.1),
        ),
    ]


def count_docs_in_index(os_store: OpensearchVectorStore) -> int:
    """Refresh indices and return the count of documents in the index."""
    os_store.client._os_client.indices.refresh(index=os_store.client._index)
    count = os_store.client._os_client.count(index=os_store.client._index)
    return count["count"]


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
def test_functionality(
    os_store: OpensearchVectorStore, node_embeddings: List[TextNode]
) -> None:
    # add
    assert len(os_store.add(node_embeddings)) == len(node_embeddings)
    # query
    exp_node = node_embeddings[3]
    query = VectorStoreQuery(query_embedding=exp_node.embedding, similarity_top_k=1)
    query_result = os_store.query(query)
    assert query_result.nodes
    assert query_result.nodes[0].get_content() == exp_node.text
    # delete one node using its associated doc_id
    os_store.delete("test-1")
    assert count_docs_in_index(os_store) == len(node_embeddings) - 1


@pytest.mark.asyncio()
@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
async def test_async_functionality(
    os_store: OpensearchVectorStore, node_embeddings: List[TextNode]
) -> None:
    # add
    assert len(await os_store.async_add(node_embeddings)) == len(node_embeddings)
    # query
    exp_node = node_embeddings[3]
    query = VectorStoreQuery(query_embedding=exp_node.embedding, similarity_top_k=1)
    query_result = await os_store.aquery(query)
    assert query_result.nodes
    assert query_result.nodes[0].get_content() == exp_node.text
    # delete one node using its associated doc_id
    await os_store.adelete("test-1")
    assert count_docs_in_index(os_store) == len(node_embeddings) - 1


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
def test_delete_nodes(
    os_store: OpensearchVectorStore, node_embeddings_2: List[TextNode]
):
    os_store.add(node_embeddings_2)

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.5), similarity_top_k=10)

    # test deleting nothing
    os_store.delete_nodes()
    time.sleep(1)
    res = os_store.query(q)
    assert all(i in res.ids for i in ["aaa", "bbb", "ccc"])

    # test deleting element that doesn't exist
    os_store.delete_nodes(["asdf"])
    time.sleep(1)
    res = os_store.query(q)
    assert all(i in res.ids for i in ["aaa", "bbb", "ccc"])

    # test deleting list
    os_store.delete_nodes(["aaa", "bbb"])
    time.sleep(1)
    res = os_store.query(q)
    assert all(i not in res.ids for i in ["aaa", "bbb"])
    assert "ccc" in res.ids


@pytest.mark.asyncio()
@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
async def test_adelete_nodes(
    os_store: OpensearchVectorStore, node_embeddings_2: List[TextNode]
):
    await os_store.async_add(node_embeddings_2)

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.5), similarity_top_k=10)

    # test deleting nothing
    await os_store.adelete_nodes()
    time.sleep(1)
    res = await os_store.aquery(q)
    assert all(i in res.ids for i in ["aaa", "bbb", "ccc"])

    # test deleting element that doesn't exist
    await os_store.adelete_nodes(["asdf"])
    time.sleep(1)
    res = await os_store.aquery(q)
    assert all(i in res.ids for i in ["aaa", "bbb", "ccc"])

    # test deleting list
    await os_store.adelete_nodes(["aaa", "bbb"])
    time.sleep(1)
    res = await os_store.aquery(q)
    assert all(i not in res.ids for i in ["aaa", "bbb"])
    assert "ccc" in res.ids


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
def test_delete_nodes_metadata(
    os_store: OpensearchVectorStore, node_embeddings_2: List[TextNode]
) -> None:
    os_store.add(node_embeddings_2)

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.5), similarity_top_k=10)

    # test deleting multiple IDs but only one satisfies filter
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key",
                value="test_value",
                operator=FilterOperator.EQ,
            )
        ]
    )
    os_store.delete_nodes(["aaa", "bbb"], filters=filters)
    time.sleep(1)
    res = os_store.query(q)
    assert all(i in res.ids for i in ["aaa", "ccc", "ddd"])
    assert "bbb" not in res.ids

    # test deleting one ID which satisfies the filter
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_num",
                value=1,
                operator=FilterOperator.EQ,
            )
        ]
    )
    os_store.delete_nodes(["aaa"], filters=filters)
    time.sleep(1)
    res = os_store.query(q)
    assert all(i not in res.ids for i in ["bbb", "aaa"])
    assert all(i in res.ids for i in ["ccc", "ddd"])

    # test deleting one ID which doesn't satisfy the filter
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_num",
                value="1",
                operator=FilterOperator.EQ,
            )
        ]
    )
    os_store.delete_nodes(["ccc"], filters=filters)
    time.sleep(1)
    res = os_store.query(q)
    assert all(i not in res.ids for i in ["bbb", "aaa"])
    assert all(i in res.ids for i in ["ccc", "ddd"])

    # test deleting purely based on filters
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key_2",
                value="test_val_2",
                operator=FilterOperator.EQ,
            )
        ]
    )
    os_store.delete_nodes(filters=filters)
    time.sleep(1)
    res = os_store.query(q)
    assert all(i not in res.ids for i in ["bbb", "aaa", "ddd"])
    assert "ccc" in res.ids


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
def test_clear(
    os_store: OpensearchVectorStore, node_embeddings_2: List[TextNode]
) -> None:
    os_store.add(node_embeddings_2)

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.5), similarity_top_k=10)
    res = os_store.query(q)
    assert all(i in res.ids for i in ["bbb", "aaa", "ddd", "ccc"])

    os_store.clear()

    time.sleep(1)

    res = os_store.query(q)
    assert all(i not in res.ids for i in ["bbb", "aaa", "ddd", "ccc"])
    assert len(res.ids) == 0


@pytest.mark.asyncio()
@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
async def test_aclear(
    os_store: OpensearchVectorStore, node_embeddings_2: List[TextNode]
) -> None:
    await os_store.async_add(node_embeddings_2)

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.5), similarity_top_k=10)
    res = await os_store.aquery(q)
    assert all(i in res.ids for i in ["bbb", "aaa", "ddd", "ccc"])

    await os_store.aclear()

    time.sleep(1)

    res = await os_store.aquery(q)
    assert all(i not in res.ids for i in ["bbb", "aaa", "ddd", "ccc"])
    assert len(res.ids) == 0


@pytest.fixture()
def insert_document(os_store: OpensearchVectorStore):
    """Factory to insert a document with custom metadata into the OpensearchVectorStore."""

    def _insert_document(doc_id: str, metadata: dict):
        """Helper function to insert a document with custom metadata."""
        os_store.add(
            [
                TextNode(
                    id_=doc_id,
                    text="Lorem Ipsum",
                    metadata=metadata,
                    embedding=[0.1, 0.2, 0.3],
                )
            ]
        )

    return _insert_document


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
@pytest.mark.parametrize("operator", [FilterOperator.EQ, FilterOperator.NE])
@pytest.mark.parametrize(
    ("key", "value", "false_value"),
    [
        ("author", "John Doe", "Doe John"),
        ("created_at", "2019-03-23T21:34:46+00:00", "2020-03-23T21:34:46+00:00"),
        ("directory", "parent/sub_dir", "parent"),
        ("page", 42, 43),
    ],
)
def test_filter_eq(
    os_store: OpensearchVectorStore,
    insert_document,
    operator: FilterOperator,
    key: str,
    value,
    false_value,
):
    """Test that OpensearchVectorStore correctly applies FilterOperator.EQ/NE in filters."""
    for meta, id_ in [
        ({key: value}, "match"),
        ({key: false_value}, "nomatch1"),
        ({}, "nomatch2"),
    ]:
        insert_document(doc_id=id_, metadata=meta)

    query = _get_sample_vector_store_query(
        filters=MetadataFilters(
            filters=[MetadataFilter(key=key, value=value, operator=operator)]
        )
    )
    query_result = os_store.query(query)

    doc_ids = {node.id_ for node in query_result.nodes}
    if operator == FilterOperator.EQ:
        assert doc_ids == {"match"}
    else:  # FilterOperator.NE
        assert doc_ids == {"nomatch1", "nomatch2"}


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
@pytest.mark.parametrize(
    ("operator", "exp_doc_ids"),
    [
        (FilterOperator.GT, {"page3", "page4"}),
        (FilterOperator.GTE, {"page2", "page3", "page4"}),
        (FilterOperator.LT, {"page1"}),
        (FilterOperator.LTE, {"page1", "page2"}),
    ],
)
def test_filter_range_number(
    os_store: OpensearchVectorStore,
    insert_document,
    operator: FilterOperator,
    exp_doc_ids: set,
):
    """Test that OpensearchVectorStore correctly applies FilterOperator.GT/GTE/LT/LTE in filters for numbers."""
    for i in range(1, 5):
        insert_document(doc_id=f"page{i}", metadata={"page": i})
    insert_document(doc_id="nomatch", metadata={})

    query = _get_sample_vector_store_query(
        filters=MetadataFilters(
            filters=[MetadataFilter(key="page", value=2, operator=operator)]
        )
    )
    query_result = os_store.query(query)

    doc_ids = {node.id_ for node in query_result.nodes}
    assert doc_ids == exp_doc_ids


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
@pytest.mark.parametrize(
    ("operator", "exp_doc_ids"),
    [
        (FilterOperator.GT, {"date3", "date4"}),
        (FilterOperator.GTE, {"date2", "date3", "date4"}),
        (FilterOperator.LT, {"date1"}),
        (FilterOperator.LTE, {"date1", "date2"}),
    ],
)
def test_filter_range_datetime(
    os_store: OpensearchVectorStore,
    insert_document,
    operator: FilterOperator,
    exp_doc_ids: set,
):
    """Test that OpensearchVectorStore correctly applies FilterOperator.GT/GTE/LT/LTE in filters for datetime."""
    dt = datetime.now()
    for i in range(1, 5):
        insert_document(
            doc_id=f"date{i}", metadata={"created_at": dt.replace(second=i).isoformat()}
        )
    insert_document(doc_id="nomatch", metadata={})

    query = _get_sample_vector_store_query(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="created_at",
                    value=dt.replace(second=2).isoformat(),
                    operator=operator,
                )
            ]
        )
    )
    query_result = os_store.query(query)

    doc_ids = {node.id_ for node in query_result.nodes}
    assert doc_ids == exp_doc_ids


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
@pytest.mark.parametrize(
    ("operator", "exp_doc_ids"),
    [
        (FilterOperator.IN, {"match1", "match2"}),
        (FilterOperator.ANY, {"match1", "match2"}),
        (FilterOperator.NIN, {"nomatch"}),
    ],
)
@pytest.mark.parametrize("value", [["product"], ["accounting", "product"]])
def test_filter_in(
    os_store: OpensearchVectorStore,
    insert_document,
    operator: FilterOperator,
    exp_doc_ids: Set[str],
    value: List[str],
):
    """Test that OpensearchVectorStore correctly applies FilterOperator.IN/ANY/NIN in filters."""
    for metadata, id_ in [
        ({"category": ["product", "management"]}, "match1"),
        ({"category": ["product", "marketing"]}, "match2"),
        ({"category": ["management"]}, "nomatch"),
    ]:
        insert_document(doc_id=id_, metadata=metadata)

    query = _get_sample_vector_store_query(
        filters=MetadataFilters(
            filters=[MetadataFilter(key="category", value=value, operator=operator)]
        )
    )
    query_result = os_store.query(query)

    doc_ids = {node.id_ for node in query_result.nodes}
    assert doc_ids == exp_doc_ids


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
def test_filter_all(
    os_store: OpensearchVectorStore,
    insert_document,
):
    """Test that OpensearchVectorStore correctly applies FilterOperator.ALL in filters."""
    for metadata, id_ in [
        ({"category": ["product", "management", "marketing"]}, "match1"),
        ({"category": ["product", "marketing"]}, "match2"),
        ({"category": ["product", "management"]}, "nomatch"),
    ]:
        insert_document(doc_id=id_, metadata=metadata)

    query = _get_sample_vector_store_query(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category",
                    value=["product", "marketing"],
                    operator=FilterOperator.ALL,
                )
            ]
        )
    )
    query_result = os_store.query(query)

    doc_ids = {node.id_ for node in query_result.nodes}
    assert doc_ids == {"match1", "match2"}


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
def test_filter_text_match(
    os_store: OpensearchVectorStore,
    insert_document,
):
    """Test that OpensearchVectorStore correctly applies FilterOperator.TEXT_MATCH in filters. Also tests that
    fuzzy matching works as intended.
    """
    for metadata, id_ in [
        ({"name": "John Doe"}, "match1"),
        ({"name": "Doe John Johnson"}, "match2"),
        ({"name": "Johnny Doe"}, "match3"),
        ({"name": "Mary Sue"}, "nomatch"),
    ]:
        insert_document(doc_id=id_, metadata=metadata)

    query = _get_sample_vector_store_query(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="name", value="John Doe", operator=FilterOperator.TEXT_MATCH
                )
            ]
        )
    )
    query_result = os_store.query(query)

    doc_ids = {node.id_ for node in query_result.nodes}
    assert doc_ids == {"match1", "match2", "match3"}


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
def test_filter_contains(os_store: OpensearchVectorStore, insert_document):
    """Test that OpensearchVectorStore correctly applies FilterOperator.CONTAINS in filters. Should only match
    exact substring matches.
    """
    for metadata, id_ in [
        ({"name": "John Doe"}, "match1"),
        ({"name": "Johnny Doe"}, "match2"),
        ({"name": "Jon Doe"}, "nomatch"),
    ]:
        insert_document(doc_id=id_, metadata=metadata)

    query = _get_sample_vector_store_query(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="name", value="ohn", operator=FilterOperator.CONTAINS
                )
            ]
        )
    )
    query_result = os_store.query(query)

    doc_ids = {node.id_ for node in query_result.nodes}
    assert doc_ids == {"match1", "match2"}


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
@pytest.mark.parametrize(
    ("filters", "exp_match_ids"),
    [
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(key="page", value=42),
                    MetadataFilters(
                        filters=[
                            MetadataFilter(
                                key="status",
                                value="published",
                                operator=FilterOperator.EQ,
                            ),
                            MetadataFilter(
                                key="category",
                                value=["group1", "group2"],
                                operator=FilterOperator.ANY,
                            ),
                        ],
                        condition=FilterCondition.OR,
                    ),
                ],
                condition=FilterCondition.AND,
            ),
            {"doc_in_category", "doc_published"},
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(key="page", value=42, operator=FilterOperator.GT),
                    MetadataFilter(key="page", value=45, operator=FilterOperator.LT),
                ],
            ),
            {"page43", "page44"},
        ),
    ],
)
def test_filter_nested(
    os_store: OpensearchVectorStore,
    insert_document,
    filters: MetadataFilters,
    exp_match_ids: Set[str],
):
    """Test that OpensearchVectorStore correctly applies nested filters."""
    for metadata, id_ in [
        (
            {"category": ["group1", "group3"], "status": "in_review", "page": 42},
            "doc_in_category",
        ),
        (
            {"category": ["group3", "group4"], "status": "in_review", "page": 42},
            "nomatch1",
        ),
        (
            {"category": ["group3", "group4"], "status": "published", "page": 42},
            "doc_published",
        ),
    ]:
        insert_document(doc_id=id_, metadata=metadata)
    for i in range(43, 46):
        insert_document(doc_id=f"page{i}", metadata={"page": i})
    insert_document(doc_id="nomatch2", metadata={})

    query = _get_sample_vector_store_query(filters=filters)
    query_result = os_store.query(query)

    doc_ids = {node.id_ for node in query_result.nodes}
    assert doc_ids == exp_match_ids
