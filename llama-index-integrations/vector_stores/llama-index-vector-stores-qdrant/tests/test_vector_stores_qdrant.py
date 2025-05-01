import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointsList,
    PointStruct,
    Filter,
)
from unittest.mock import MagicMock

from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode


def test_class():
    names_of_base_classes = [b.__name__ for b in QdrantVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_delete__and_get_nodes(vector_store: QdrantVectorStore) -> None:
    vector_store.delete_nodes(node_ids=["11111111-1111-1111-1111-111111111111"])

    existing_nodes = vector_store.get_nodes(
        node_ids=[
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            "33333333-3333-3333-3333-333333333333",
        ]
    )
    assert len(existing_nodes) == 2


def test_clear(vector_store: QdrantVectorStore) -> None:
    vector_store.clear()
    with pytest.raises(ValueError, match="Collection test not found"):
        vector_store.get_nodes(
            node_ids=[
                "11111111-1111-1111-1111-111111111111",
                "22222222-2222-2222-2222-222222222222",
                "33333333-3333-3333-3333-333333333333",
            ]
        )


@pytest.mark.asyncio
async def test_adelete_and_aget(vector_store: QdrantVectorStore) -> None:
    await vector_store.adelete_nodes(node_ids=["11111111-1111-1111-1111-111111111111"])

    existing_nodes = await vector_store.aget_nodes(
        node_ids=[
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            "33333333-3333-3333-3333-333333333333",
        ]
    )
    assert len(existing_nodes) == 2


@pytest.mark.asyncio
async def test_aclear(vector_store: QdrantVectorStore) -> None:
    await vector_store.aclear()
    with pytest.raises(ValueError, match="Collection test not found"):
        await vector_store.aget_nodes(
            node_ids=[
                "11111111-1111-1111-1111-111111111111",
                "22222222-2222-2222-2222-222222222222",
                "33333333-3333-3333-3333-333333333333",
            ]
        )


def test_parse_query_result(vector_store: QdrantVectorStore) -> None:
    payload = {
        "text": "Hello, world!",
    }

    vector_dict = {
        "": [1, 2, 3],
    }

    # test vector name is empty (default)
    points = PointsList(points=[PointStruct(id=1, vector=vector_dict, payload=payload)])

    results = vector_store.parse_to_query_result(list(points.points))

    assert len(results.nodes) == 1
    assert results.nodes[0].embedding == [1, 2, 3]

    # test vector name is not empty
    vector_dict = {
        "text-dense": [1, 2, 3],
    }

    points = PointsList(points=[PointStruct(id=1, vector=vector_dict, payload=payload)])

    vector_store.dense_vector_name = "text-dense"
    results = vector_store.parse_to_query_result(list(points.points))

    assert len(results.nodes) == 1
    assert results.nodes[0].embedding == [1, 2, 3]


@pytest.mark.asyncio
async def test_get_with_embedding(vector_store: QdrantVectorStore) -> None:
    existing_nodes = await vector_store.aget_nodes(
        node_ids=[
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            "33333333-3333-3333-3333-333333333333",
        ]
    )

    assert all(node.embedding is not None for node in existing_nodes)


def test_filter_conditions():
    """Test AND, OR, and NOT filter conditions."""
    from llama_index.core.vector_stores.types import (
        MetadataFilter,
        MetadataFilters,
        FilterCondition,
        FilterOperator,
    )

    # Create a mock Qdrant client
    mock_client = MagicMock(spec=QdrantClient)
    vector_store = QdrantVectorStore(
        collection_name="test_collection",
        client=mock_client,
    )

    # Test AND condition
    and_filter = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="books", operator=FilterOperator.EQ),
            MetadataFilter(key="price", value=10, operator=FilterOperator.GT),
        ],
        condition=FilterCondition.AND,
    )
    filter_and = vector_store._build_subfilter(and_filter)
    assert filter_and.must is not None
    assert len(filter_and.must) == 2
    assert filter_and.must[0].key == "category"
    assert filter_and.must[0].match.value == "books"
    assert filter_and.must[1].key == "price"
    assert filter_and.must[1].range.gt == 10

    # Test OR condition
    or_filter = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="books", operator=FilterOperator.EQ),
            MetadataFilter(
                key="category", value="electronics", operator=FilterOperator.EQ
            ),
        ],
        condition=FilterCondition.OR,
    )
    filter_or = vector_store._build_subfilter(or_filter)
    assert filter_or.should is not None
    assert len(filter_or.should) == 2
    assert filter_or.should[0].key == "category"
    assert filter_or.should[0].match.value == "books"
    assert filter_or.should[1].key == "category"
    assert filter_or.should[1].match.value == "electronics"

    # Test NOT condition
    not_filter = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="books", operator=FilterOperator.EQ),
        ],
        condition="not",
    )
    filter_not = vector_store._build_subfilter(not_filter)
    assert filter_not.must_not is not None
    assert len(filter_not.must_not) == 1
    assert filter_not.must_not[0].key == "category"
    assert filter_not.must_not[0].match.value == "books"

    # Test AND with NOT condition
    and_not_filter = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="books", operator=FilterOperator.EQ),
            MetadataFilters(
                filters=[
                    MetadataFilter(key="price", value=50, operator=FilterOperator.EQ),
                ],
                condition="not",
            ),
        ],
        condition=FilterCondition.AND,
    )
    filter_and_not = vector_store._build_subfilter(and_not_filter)
    assert filter_and_not.must is not None
    assert (
        len(filter_and_not.must) == 2
    )  # One for category and one for the nested filter
    assert filter_and_not.must[0].key == "category"
    assert filter_and_not.must[0].match.value == "books"
    # The second must element is a Filter object with must_not condition
    assert isinstance(filter_and_not.must[1], Filter)
    assert filter_and_not.must[1].must_not is not None
    assert len(filter_and_not.must[1].must_not) == 1
    assert filter_and_not.must[1].must_not[0].key == "price"
    assert filter_and_not.must[1].must_not[0].match.value == 50


def test_hybrid_vector_store_query(hybrid_vector_store: QdrantVectorStore) -> None:
    query = VectorStoreQuery(
        query_embedding=[0.0, 0.0],
        query_str="test1",
        similarity_top_k=1,
        sparse_top_k=1,
        hybrid_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )
    results = hybrid_vector_store.query(query)
    assert len(results.nodes) == 2

    # disable hybrid, and it should still work
    hybrid_vector_store.enable_hybrid = False
    query.mode = VectorStoreQueryMode.DEFAULT
    results = hybrid_vector_store.query(query)
    assert len(results.nodes) == 1
