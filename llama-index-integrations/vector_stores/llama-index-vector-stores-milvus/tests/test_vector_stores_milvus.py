from typing import List, Dict

import pytest

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    MetadataFilter,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.base import _to_milvus_filter
from llama_index.vector_stores.milvus.utils import (
    ScalarMetadataFilter,
    ScalarMetadataFilters,
    FilterOperatorFunction,
    BaseSparseEmbeddingFunction,
)
import pytest_asyncio

TEST_URI = "./milvus_test.db"


class MockSparseEmbeddingFunction(BaseSparseEmbeddingFunction):
    def encode_queries(self, queries: List[str]) -> List[Dict[int, float]]:
        return [{1: 0.5, 2: 0.3}] * len(queries)

    def encode_documents(self, documents: List[str]) -> List[Dict[int, float]]:
        return [{1: 0.5, 2: 0.3}] * len(documents)


def test_class():
    names_of_base_classes = [b.__name__ for b in MilvusVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_to_milvus_filter_with_scalar_filters():
    filters = None
    scalar_filters = ScalarMetadataFilters(
        filters=[ScalarMetadataFilter(key="a", value=1)]
    )
    expr = _to_milvus_filter(filters, scalar_filters.to_dict())
    assert expr == "ARRAY_CONTAINS(a, 1)"

    scalar_filters = ScalarMetadataFilters(
        filters=[
            ScalarMetadataFilter(
                key="a", value=1, operator=FilterOperatorFunction.NARRAY_CONTAINS
            )
        ]
    )
    expr = _to_milvus_filter(filters, scalar_filters.to_dict())
    assert expr == "not ARRAY_CONTAINS(a, 1)"

    scalar_filters = ScalarMetadataFilters(
        filters=[
            ScalarMetadataFilter(
                key="a", value="b", operator=FilterOperatorFunction.NARRAY_CONTAINS
            ),
            ScalarMetadataFilter(
                key="c", value=2, operator=FilterOperatorFunction.ARRAY_LENGTH
            ),
        ]
    )
    expr = _to_milvus_filter(filters, scalar_filters.to_dict())
    assert expr == "(not ARRAY_CONTAINS(a, 'b') and ARRAY_LENGTH(c) == 2)"


def test_to_milvus_filter_with_various_operators():
    filters = MetadataFilters(filters=[MetadataFilter(key="a", value=1)])
    expr = _to_milvus_filter(filters)
    assert expr == "a == 1"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.NE)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a != 1"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.GT)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a > 1"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.GTE)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a >= 1"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.LT)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a < 1"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.LTE)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a <= 1"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=[1, 2, 3], operator=FilterOperator.IN)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a in [1, 2, 3]"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=[1, 2, 3], operator=FilterOperator.NIN)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a not in [1, 2, 3]"

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="a", value="substring", operator=FilterOperator.TEXT_MATCH
            )
        ]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a like 'substring%'"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.CONTAINS)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "array_contains(a, 1)"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=[1, 2], operator=FilterOperator.ANY)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "array_contains_any(a, [1, 2])"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=[1, 2], operator=FilterOperator.ALL)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "array_contains_all(a, [1, 2])"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=None, operator=FilterOperator.IS_EMPTY)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "array_length(a) == 0"


def test_to_milvus_filter_with_string_value():
    filters = MetadataFilters(filters=[MetadataFilter(key="a", value="hello")])
    expr = _to_milvus_filter(filters)
    assert expr == "a == 'hello'"

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value="hello", operator=FilterOperator.CONTAINS)
        ]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "array_contains(a, 'hello')"

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=["you", "me"], operator=FilterOperator.ANY)
        ]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "array_contains_any(a, ['you', 'me'])"

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=["you", "me"], operator=FilterOperator.ALL)
        ]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "array_contains_all(a, ['you', 'me'])"


def test_to_milvus_filter_with_multiple_filters():
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.GTE),
            MetadataFilter(key="a", value=10, operator=FilterOperator.LTE),
        ],
        condition=FilterCondition.AND,
    )
    expr = _to_milvus_filter(filters)
    assert expr == "(a >= 1 and a <= 10)"

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.LT),
            MetadataFilter(key="a", value=10, operator=FilterOperator.GT),
        ],
        condition=FilterCondition.OR,
    )
    expr = _to_milvus_filter(filters)
    assert expr == "(a < 1 or a > 10)"


def test_milvus_filter_with_nested_filters():
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.EQ),
            MetadataFilters(
                filters=[
                    MetadataFilter(key="b", value=2, operator=FilterOperator.EQ),
                    MetadataFilter(key="c", value=3, operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )
    expr = _to_milvus_filter(filters)
    assert expr == "(a == 1 and (b == 2 or c == 3))"


@pytest.mark.asyncio()
class TestMilvusAsync:
    @pytest_asyncio.fixture
    def vector_store(self, event_loop) -> MilvusVectorStore:
        yield MilvusVectorStore(
            uri=TEST_URI,
            dim=64,
            collection_name="test_collection",
            embedding_field="embedding",
            id_field="id",
            similarity_metric="COSINE",
            consistency_level="Strong",
            overwrite=True,
        )

    async def test_milvus_delete(self, vector_store: MilvusVectorStore, event_loop):
        node1 = TextNode(
            id_="n1",
            text="n1_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
        )
        node2 = TextNode(
            id_="n2",
            text="n2_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
        )
        await vector_store.async_add([node1, node2])
        await vector_store.adelete(ref_doc_id="n2")
        query_res = await vector_store.aclient.query(
            "test_collection", output_fields=["count(*)"]
        )
        assert query_res[0]["count(*)"] == 1
        await vector_store.adelete(ref_doc_id="n3")
        query_res = await vector_store.aclient.query(
            "test_collection", output_fields=["count(*)"]
        )
        assert query_res[0]["count(*)"] == 0

    async def test_milvus_delete_nodes(
        self, vector_store: MilvusVectorStore, event_loop
    ):
        node1 = TextNode(
            id_="n1",
            text="n1_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
        )
        node2 = TextNode(
            id_="n2",
            text="n2_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
        )
        await vector_store.async_add([node1, node2])
        await vector_store.adelete_nodes(node_ids=["n1"])
        query_res = await vector_store.aclient.query(
            "test_collection", output_fields=["count(*)"]
        )
        assert query_res[0]["count(*)"] == 1
        await vector_store.adelete_nodes(node_ids=["n2"])
        query_res = await vector_store.aclient.query(
            "test_collection", output_fields=["count(*)"]
        )
        assert query_res[0]["count(*)"] == 0

    async def test_milvus_clear(self, vector_store: MilvusVectorStore, event_loop):
        await vector_store.aclear()
        assert not vector_store.client.has_collection("test_collection")

    async def test_get_nodes(self, vector_store: MilvusVectorStore, event_loop):
        node1 = TextNode(
            id_="n1",
            text="n1_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
        )
        node2 = TextNode(
            id_="n2",
            text="n2_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
        )
        await vector_store.async_add([node1, node2])
        nodes = await vector_store.aget_nodes(node_ids=["n1"])
        assert nodes[0] == node1
        nodes = await vector_store.aget_nodes(node_ids=["n1", "n2"])
        assert node1 in nodes and node2 in nodes and len(nodes) == 2

    async def test_query_default_mode(
        self, vector_store: MilvusVectorStore, event_loop
    ):
        node1 = TextNode(
            id_="n1",
            text="n1_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
        )
        node2 = TextNode(
            id_="n2",
            text="n2_text",
            embedding=[-0.5] * 64,  # opposite direction of node1's embedding
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
        )
        await vector_store.async_add([node1, node2])

        query = VectorStoreQuery(query_embedding=[0.5] * 64, similarity_top_k=1)
        result = await vector_store.aquery(query=query)
        assert len(result.nodes) == 1
        assert result.nodes[0].id_ == "n1"
        assert result.nodes[0].text == "n1_text"

        query = VectorStoreQuery(query_embedding=[-0.5] * 64, similarity_top_k=1)
        result = await vector_store.aquery(query=query)
        assert len(result.nodes) == 1
        assert result.nodes[0].id_ == "n2"
        assert result.nodes[0].text == "n2_text"

    async def test_query_mmr_mode(self, vector_store: MilvusVectorStore, event_loop):
        node1 = TextNode(
            id_="n1",
            text="n1_text",
            embedding=[0.5] * 63 + [0.0],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
        )
        node2 = TextNode(
            id_="n2",
            text="n2_text",
            embedding=[0.5] * 63 + [0.2],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
        )
        node3 = TextNode(
            id_="n3",
            text="n3_text",
            embedding=[0.5] * 63 + [0.4],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n4")},
        )
        await vector_store.async_add([node1, node2, node3])
        query = VectorStoreQuery(
            query_embedding=[0.5] * 64,
            similarity_top_k=2,
            mode=VectorStoreQueryMode.MMR,
        )
        result = await vector_store.aquery(query=query, mmr_prefetch_k=3)
        assert len(result.nodes) == 2
        assert result.nodes[0].id_ == "n3"
        assert result.nodes[0].text == "n3_text"
        assert result.nodes[1].id_ == "n2"
        assert result.nodes[1].text == "n2_text"

    async def test_query_hybrid_mode(self, event_loop):
        vector_store = MilvusVectorStore(
            uri=TEST_URI,
            dim=64,
            collection_name="test_collection",
            overwrite=True,
            enable_sparse=True,
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": 60},
            sparse_embedding_function=MockSparseEmbeddingFunction(),
            consistency_level="Strong",
        )
        node1 = TextNode(
            id_="n1",
            text="n1_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
        )
        node2 = TextNode(
            id_="n2",
            text="n2_text",
            embedding=[-0.5] * 64,  # opposite direction of node1's embedding
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
        )
        await vector_store.async_add([node1, node2])
        query = VectorStoreQuery(
            query_embedding=[0.5] * 64,
            query_str="mock_str",
            similarity_top_k=1,
            mode=VectorStoreQueryMode.HYBRID,
        )
        result = await vector_store.aquery(query=query)
        assert len(result.nodes) == 1
        assert result.nodes[0].id_ == "n1"
        assert result.nodes[0].text == "n1_text"


class TestMilvusSync:
    @pytest.fixture()
    def vector_store(self) -> MilvusVectorStore:
        return MilvusVectorStore(
            uri=TEST_URI,
            dim=64,
            collection_name="test_collection",
            embedding_field="embedding",
            id_field="id",
            similarity_metric="COSINE",
            consistency_level="Strong",
            overwrite=True,
        )

    def test_milvus_delete(self, vector_store: MilvusVectorStore):
        node1 = TextNode(
            id_="n1",
            text="n1_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
        )
        node2 = TextNode(
            id_="n2",
            text="n2_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
        )
        vector_store.add([node1, node2])
        vector_store.delete(ref_doc_id="n2")
        row_count = vector_store.client.query(
            "test_collection", output_fields=["count(*)"]
        )[0]["count(*)"]
        assert row_count == 1
        vector_store.delete(ref_doc_id="n3")
        row_count = vector_store.client.query(
            "test_collection", output_fields=["count(*)"]
        )[0]["count(*)"]
        assert row_count == 0

    def test_milvus_delete_nodes(self, vector_store: MilvusVectorStore):
        node1 = TextNode(
            id_="n1",
            text="n1_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
        )
        node2 = TextNode(
            id_="n2",
            text="n2_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
        )
        vector_store.add([node1, node2])
        vector_store.delete_nodes(node_ids=["n1"])
        row_count = vector_store.client.query(
            "test_collection", output_fields=["count(*)"]
        )[0]["count(*)"]
        assert row_count == 1
        vector_store.delete_nodes(node_ids=["n2"])
        row_count = vector_store.client.query(
            "test_collection", output_fields=["count(*)"]
        )[0]["count(*)"]
        assert row_count == 0

    def test_milvus_clear(self, vector_store: MilvusVectorStore):
        vector_store.clear()
        assert not vector_store.client.has_collection("test_collection")

    def test_get_nodes(self, vector_store: MilvusVectorStore):
        node1 = TextNode(
            id_="n1",
            text="n1_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
        )
        node2 = TextNode(
            id_="n2",
            text="n2_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
        )
        vector_store.add([node1, node2])
        nodes = vector_store.get_nodes(node_ids=["n1"])
        assert nodes[0] == node1
        nodes = vector_store.get_nodes(node_ids=["n1", "n2"])
        assert node1 in nodes and node2 in nodes and len(nodes) == 2

    def test_query_default_mode(self, vector_store: MilvusVectorStore):
        node1 = TextNode(
            id_="n1",
            text="n1_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
        )
        node2 = TextNode(
            id_="n2",
            text="n2_text",
            embedding=[-0.5] * 64,  # opposite direction of node1's embedding
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
        )
        vector_store.add([node1, node2])

        query = VectorStoreQuery(query_embedding=[0.5] * 64, similarity_top_k=1)
        result = vector_store.query(query=query)
        assert len(result.nodes) == 1
        assert result.nodes[0].id_ == "n1"
        assert result.nodes[0].text == "n1_text"

        query = VectorStoreQuery(query_embedding=[-0.5] * 64, similarity_top_k=1)
        result = vector_store.query(query=query)
        assert len(result.nodes) == 1
        assert result.nodes[0].id_ == "n2"
        assert result.nodes[0].text == "n2_text"

    def test_query_mmr_mode(self, vector_store: MilvusVectorStore):
        node1 = TextNode(
            id_="n1",
            text="n1_text",
            embedding=[0.5] * 63 + [0.0],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
        )
        node2 = TextNode(
            id_="n2",
            text="n2_text",
            embedding=[0.5] * 63 + [0.2],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
        )
        node3 = TextNode(
            id_="n3",
            text="n3_text",
            embedding=[0.5] * 63 + [0.4],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n4")},
        )
        vector_store.add([node1, node2, node3])
        query = VectorStoreQuery(
            query_embedding=[0.5] * 64,
            similarity_top_k=2,
            mode=VectorStoreQueryMode.MMR,
        )
        result = vector_store.query(query=query, mmr_prefetch_k=3)
        assert len(result.nodes) == 2
        assert result.nodes[0].id_ == "n3"
        assert result.nodes[0].text == "n3_text"
        assert result.nodes[1].id_ == "n2"
        assert result.nodes[1].text == "n2_text"

    def test_query_hybrid_mode(self):
        vector_store = MilvusVectorStore(
            uri=TEST_URI,
            dim=64,
            collection_name="test_collection",
            overwrite=True,
            enable_sparse=True,
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": 60},
            sparse_embedding_function=MockSparseEmbeddingFunction(),
            consistency_level="Strong",
        )
        node1 = TextNode(
            id_="n1",
            text="n1_text",
            embedding=[0.5] * 64,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
        )
        node2 = TextNode(
            id_="n2",
            text="n2_text",
            embedding=[-0.5] * 64,  # opposite direction of node1's embedding
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
        )
        vector_store.add([node1, node2])
        query = VectorStoreQuery(
            query_embedding=[0.5] * 64,
            query_str="mock_str",
            similarity_top_k=1,
            mode=VectorStoreQueryMode.HYBRID,
        )
        result = vector_store.query(query=query)
        assert len(result.nodes) == 1
        assert result.nodes[0].id_ == "n1"
        assert result.nodes[0].text == "n1_text"
