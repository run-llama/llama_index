import pytest
import os
import requests
import httpx

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointsList,
    PointStruct,
    Filter,
)
from unittest.mock import MagicMock

from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    MetadataFilters,
    MetadataFilter,
    FilterCondition,
    FilterOperator,
)

requires_qdrant_cluster = pytest.mark.skipif(
    not os.getenv("QDRANT_CLUSTER_URL"),
    reason="Qdrant cluster not available in CI",
)


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


def test_filters_with_types(vector_store: QdrantVectorStore) -> None:
    results = vector_store.get_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="some_key", value=[1, 2], operator=FilterOperator.IN)
            ]
        )
    )
    assert len(results) == 2

    results = vector_store.get_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="some_key", value=[1, 2], operator=FilterOperator.NIN
                )
            ]
        )
    )
    assert len(results) == 1

    results = vector_store.get_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="some_key", value=["3"], operator=FilterOperator.IN)
            ]
        )
    )
    assert len(results) == 1


def test_hybrid_vector_store_query(
    hybrid_vector_store: QdrantVectorStore,
) -> None:
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


@pytest.mark.asyncio
@requires_qdrant_cluster
async def test_shard_vector_store_async(
    shard_vector_store: QdrantVectorStore,
) -> None:
    """
    Validate that LlamaIndex's QdrantVectorStore custom sharding + metadata filtering
    behaves as expected.

    Setup (see fixture below):
      - Three nodes, each assigned to a different *custom* shard (1, 2, 3).
      - Metadata key "some_key" set to 1, 2, and "3" respectively (note: "3" is a string).
    """
    # 1) Sanity check: without filters or shard restriction, we should see all 3 nodes.
    results = await shard_vector_store.aget_nodes()
    assert len(results) == 3

    # 2) Query *only* shard 3, but with IN filter for values [1, 2].
    #    Because shard 3 contains the node with metadata "some_key" == "3" (string),
    #    there should be no matches for the integer values 1 or 2 in that shard.
    results = await shard_vector_store.aget_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="some_key", value=[1, 2], operator=FilterOperator.IN)
            ]
        ),
        shard_identifier=3,
    )

    assert len(results) == 0

    # 3) Still on shard 3, use NIN (NOT IN) for [1, 2].
    #    The only node in shard 3 has "some_key" == "3" (string), which is not 1 or 2.
    #    So we expect exactly 1 match.
    results = await shard_vector_store.aget_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="some_key", value=[1, 2], operator=FilterOperator.NIN
                )
            ]
        ),
        shard_identifier=3,
    )
    assert len(results) == 1

    # 4) Query shard 3 with IN filter for "3" (string).
    #    This should match the only node in shard 3.
    results = await shard_vector_store.aget_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="some_key", value=["3"], operator=FilterOperator.IN)
            ]
        ),
        shard_identifier=3,
    )
    assert len(results) == 1

    # 5) Delete the node in shard 3.
    #    This should remove the only node in shard 3.
    await shard_vector_store.adelete_nodes(
        shard_identifier=3,
    )

    results = await shard_vector_store.aget_nodes(
        shard_identifier=3,
    )
    assert len(results) == 0  # No nodes should remain in shard 3
    results = await shard_vector_store.aget_nodes()
    assert len(results) == 2  # Only nodes in shards 1 and 2 should remain

    # 6) Delete the node in shard 2 with ref_doc_id "test-0"
    #    This should remove the only node in shard 2.
    #    This shouldn't remove the node in shard 1 having same ref_doc_id.
    await shard_vector_store.adelete(
        ref_doc_id="test-0",
        shard_identifier=2,
    )

    results = await shard_vector_store.aget_nodes(
        shard_identifier=2,
    )
    assert len(results) == 0  # No nodes should remain in shard 2
    results = await shard_vector_store.aget_nodes()
    assert len(results) == 1  # Only nodes in shards 1 should remain


@pytest.mark.asyncio
@requires_qdrant_cluster
def test_shard_vector_store_sync(
    shard_vector_store: QdrantVectorStore,
) -> None:
    """
    Validate that LlamaIndex's QdrantVectorStore custom sharding + metadata filtering
    behaves as expected.

    Setup (see fixture below):
      - Three nodes, each assigned to a different *custom* shard (1, 2, 3).
      - Metadata key "some_key" set to 1, 2, and "3" respectively (note: "3" is a string).
    """
    # 1) Sanity check: without filters or shard restriction, we should see all 3 nodes.
    results = shard_vector_store.get_nodes()
    assert len(results) == 3

    # 2) Query *only* shard 3, but with IN filter for values [1, 2].
    #    Because shard 3 contains the node with metadata "some_key" == "3" (string),
    #    there should be no matches for the integer values 1 or 2 in that shard.
    results = shard_vector_store.get_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="some_key", value=[1, 2], operator=FilterOperator.IN)
            ]
        ),
        shard_identifier=3,
    )

    assert len(results) == 0

    # 3) Still on shard 3, use NIN (NOT IN) for [1, 2].
    #    The only node in shard 3 has "some_key" == "3" (string), which is not 1 or 2.
    #    So we expect exactly 1 match.
    results = shard_vector_store.get_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="some_key", value=[1, 2], operator=FilterOperator.NIN
                )
            ]
        ),
        shard_identifier=3,
    )
    assert len(results) == 1

    # 4) Query shard 3 with IN filter for "3" (string).
    #    This should match the only node in shard 3.
    results = shard_vector_store.get_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="some_key", value=["3"], operator=FilterOperator.IN)
            ]
        ),
        shard_identifier=3,
    )
    assert len(results) == 1

    # 5) Delete the node in shard 3.
    #    This should remove the only node in shard 3.
    shard_vector_store.delete_nodes(
        shard_identifier=3,
    )

    results = shard_vector_store.get_nodes(
        shard_identifier=3,
    )
    assert len(results) == 0  # No nodes should remain in shard 3
    results = shard_vector_store.get_nodes()
    assert len(results) == 2  # Only nodes in shards 1 and 2 should remain

    # 6) Delete the node in shard 2 with ref_doc_id "test-0"
    #    This should remove the only node in shard 2.
    #    This shouldn't remove the node in shard 1 having same ref_doc_id.
    shard_vector_store.delete(
        ref_doc_id="test-0",
        shard_identifier=2,
    )

    results = shard_vector_store.get_nodes(
        shard_identifier=2,
    )
    assert len(results) == 0  # No nodes should remain in shard 2
    results = shard_vector_store.get_nodes()
    assert len(results) == 1  # Only nodes in shards 1 should remain


@requires_qdrant_cluster
def test_validate_custom_sharding(shard_vector_store: QdrantVectorStore):
    shard_vector_store._validate_custom_sharding()


def test_generate_shard_key_selector_none(vector_store: QdrantVectorStore):
    # no selector function set
    assert vector_store._generate_shard_key_selector(None) is None
    assert vector_store._generate_shard_key_selector(5) is None


@requires_qdrant_cluster
def test_generate_shard_key_selector_custom(
    shard_vector_store: QdrantVectorStore,
):
    selector = shard_vector_store._generate_shard_key_selector(7)
    assert selector is not None
    assert selector == 1


@pytest.mark.asyncio
@requires_qdrant_cluster
async def test_shard_keys_created(shard_vector_store: QdrantVectorStore):
    # Ensure shard key configuration exists and expected number of shard keys created
    base_url = os.getenv("QDRANT_CLUSTER_URL")
    if not base_url:
        raise RuntimeError(
            "QDRANT_CLUSTER_URL environment variable must be set for direct HTTP call"
        )

    url = f"http://{base_url.rstrip('/')}/collections/{shard_vector_store.collection_name}/cluster"

    # Use async HTTP client instead of blocking requests in an async test

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        raw = resp.json()

    print(raw)
    result = raw.get("result", {})
    shard_keys = {
        shard.get("shard_key")
        for shard in result.get("local_shards", []) + result.get("remote_shards", [])
    }
    assert shard_keys, "Shard key parameter missing in collection config"
    assert len(shard_keys) == 3


@requires_qdrant_cluster
def test_shard_keys_created_sync(shard_vector_store: QdrantVectorStore):
    # Ensure shard key configuration exists and expected number of shard keys created
    # Determine base URL (env var name per instructions; fall back to QDRANT_CLUSTER_URL if typo)
    base_url = os.getenv("QDRANT_CLUSTER_URL")
    if not base_url:
        raise RuntimeError(
            "QDRANT_CLUSTER_URL environment variable must be set for direct HTTP call"
        )

    url = f"http://{base_url.rstrip('/')}/collections/{shard_vector_store.collection_name}/cluster"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    print(raw)
    result = raw.get("result", {})
    shard_keys = set()
    for shard in result.get("local_shards", []) + result.get("remote_shards", []):
        shard_keys.add(shard.get("shard_key"))
    assert shard_keys, "Shard key parameter missing in collection config"
    # Expect 3 custom shard keys based on fixture inserting 3 shards
    assert len(shard_keys) == 3


@pytest.mark.asyncio
@requires_qdrant_cluster
async def test_shard_vector_store_aquery(
    shard_vector_store: QdrantVectorStore,
):
    # Test the aquery functionality of the shard vector store
    query = VectorStoreQuery(
        query_embedding=[0.0, 0.0], query_str="test1", similarity_top_k=5
    )
    results = await shard_vector_store.aquery(query, shard_identifier=3)
    assert results is not None
    assert len(results.nodes) == 1


@requires_qdrant_cluster
def test_shard_vector_store_query(
    shard_vector_store: QdrantVectorStore,
):
    # Test the query functionality of the shard vector store
    query = VectorStoreQuery(
        query_embedding=[0.0, 0.0], query_str="test1", similarity_top_k=5
    )
    results = shard_vector_store.query(query, shard_identifier=3)
    assert results is not None
    assert len(results.nodes) == 1
