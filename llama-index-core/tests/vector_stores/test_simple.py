import unittest
from pathlib import Path
from typing import List

import pytest

from llama_index.core import VectorStoreIndex, MockEmbedding
from llama_index.core.schema import (
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
    Document,
)
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStoreQuery,
    FilterCondition,
    MetadataFilter,
    FilterOperator,
)

_NODE_ID_WEIGHT_1_RANK_A = "AF3BE6C4-5F43-4D74-B075-6B0E07900DE8"
_NODE_ID_WEIGHT_2_RANK_C = "7D9CD555-846C-445C-A9DD-F8924A01411D"
_NODE_ID_WEIGHT_3_RANK_C = "452D24AB-F185-414C-A352-590B4B9EE51B"


@pytest.fixture()
def persist_dir(tmp_path: Path):
    index = VectorStoreIndex.from_documents(
        [Document(id_="1", text="1")], embed_model=MockEmbedding(embed_dim=1)
    )
    index.storage_context.persist(str(tmp_path))
    return str(tmp_path)


def _node_embeddings_for_test() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_=_NODE_ID_WEIGHT_1_RANK_A,
            embedding=[1.0, 0.0],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
            metadata={
                "weight": 1.0,
                "rank": "a",
                "quality": ["medium", "high"],
                "identifier": "6FTR78Yun",
            },
        ),
        TextNode(
            text="lorem ipsum",
            id_=_NODE_ID_WEIGHT_2_RANK_C,
            embedding=[0.0, 1.0],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={
                "weight": 2.0,
                "rank": "c",
                "quality": ["medium"],
                "identifier": "6FTR78Ygl",
            },
        ),
        TextNode(
            text="lorem ipsum",
            id_=_NODE_ID_WEIGHT_3_RANK_C,
            embedding=[1.0, 1.0],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
            metadata={
                "weight": 3.0,
                "rank": "c",
                "quality": ["low", "medium", "high"],
                "identifier": "6FTR78Ztl",
            },
        ),
    ]


class SimpleVectorStoreTest(unittest.TestCase):
    def test_query_without_filters_returns_all_rows_sorted_by_similarity(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        query = VectorStoreQuery(query_embedding=[1.0, 1.0], similarity_top_k=3)
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertCountEqual(
            result.ids,
            [
                _NODE_ID_WEIGHT_1_RANK_A,
                _NODE_ID_WEIGHT_2_RANK_C,
                _NODE_ID_WEIGHT_3_RANK_C,
            ],
        )
        self.assertEqual(result.ids[0], _NODE_ID_WEIGHT_3_RANK_C)

    def test_query_with_filters_returns_multiple_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(filters=[ExactMatchFilter(key="rank", value="c")])
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        self.assertEqual(
            result.ids, [_NODE_ID_WEIGHT_3_RANK_C, _NODE_ID_WEIGHT_2_RANK_C]
        )

    def test_query_with_filter_applies_top_k(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(filters=[ExactMatchFilter(key="rank", value="c")])
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=1
        )
        result = simple_vector_store.query(query)
        self.assertEqual(result.ids, [_NODE_ID_WEIGHT_3_RANK_C])

    def test_query_with_filter_applies_node_id_filter(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(filters=[ExactMatchFilter(key="rank", value="c")])
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0],
            filters=filters,
            similarity_top_k=3,
            node_ids=[_NODE_ID_WEIGHT_3_RANK_C],
        )
        result = simple_vector_store.query(query)
        self.assertEqual(result.ids, [_NODE_ID_WEIGHT_3_RANK_C])

    def test_query_with_exact_filters_returns_single_match(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                ExactMatchFilter(key="rank", value="c"),
                ExactMatchFilter(key="weight", value=2.0),
            ]
        )
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
        result = simple_vector_store.query(query)
        self.assertEqual(result.ids, [_NODE_ID_WEIGHT_2_RANK_C])

    def test_query_with_contradictive_filter_returns_no_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                ExactMatchFilter(key="weight", value=2),
                ExactMatchFilter(key="weight", value=3),
            ]
        )
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertEqual(len(result.ids), 0)

    def test_query_with_filter_on_unknown_field_returns_no_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="unknown_field", value="c")]
        )
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertEqual(len(result.ids), 0)

    def test_delete_removes_document_from_query_results(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        simple_vector_store.delete("test-1")
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], similarity_top_k=2)
        result = simple_vector_store.query(query)
        self.assertEqual(
            result.ids,
            [_NODE_ID_WEIGHT_3_RANK_C, _NODE_ID_WEIGHT_1_RANK_A],
        )

    def test_query_with_filters_with_filter_condition(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        # test OR filter
        filters = MetadataFilters(
            filters=[
                ExactMatchFilter(key="rank", value="c"),
                ExactMatchFilter(key="weight", value=1.0),
            ],
            condition=FilterCondition.OR,
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        self.assertEqual(len(result.ids), 3)

        # test AND filter
        filters = MetadataFilters(
            filters=[
                ExactMatchFilter(key="rank", value="c"),
                ExactMatchFilter(key="weight", value=1.0),
            ],
            condition=FilterCondition.AND,
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        self.assertEqual(len(result.ids), 0)

    def test_query_with_equal_filter_returns_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="weight", operator=FilterOperator.EQ, value=1.0)
            ]
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertEqual(len(result.ids), 1)

    def test_query_with_notequal_filter_returns_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="weight", operator=FilterOperator.NE, value=1.0)
            ]
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertEqual(len(result.ids), 2)

    def test_query_with_greaterthan_filter_returns_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="weight", operator=FilterOperator.GT, value=1.5)
            ]
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertEqual(len(result.ids), 2)

    def test_query_with_greaterthanequal_filter_returns_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="weight", operator=FilterOperator.GTE, value=1.0)
            ]
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertEqual(len(result.ids), 3)

    def test_query_with_lessthan_filter_returns_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="weight", operator=FilterOperator.LT, value=1.1)
            ]
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        assert result.ids is not None

    def test_query_with_lessthanequal_filter_returns_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="weight", operator=FilterOperator.LTE, value=1.0)
            ]
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertEqual(len(result.ids), 1)

    def test_query_with_in_filter_returns_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="rank", operator=FilterOperator.IN, value=["a", "c"])
            ]
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertEqual(len(result.ids), 3)

    def test_query_with_notin_filter_returns_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="rank", operator=FilterOperator.NIN, value=["c"])
            ]
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertEqual(len(result.ids), 1)

    def test_query_with_contains_filter_returns_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="quality", operator=FilterOperator.CONTAINS, value="high"
                )
            ]
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertEqual(len(result.ids), 2)

    def test_query_with_textmatch_filter_returns_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="identifier",
                    operator=FilterOperator.TEXT_MATCH,
                    value="6FTR78Y",
                )
            ]
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertEqual(len(result.ids), 2)

    def test_query_with_any_filter_returns_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="quality", operator=FilterOperator.ANY, value=["high", "low"]
                )
            ]
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertEqual(len(result.ids), 2)

    def test_query_with_all_filter_returns_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="quality", operator=FilterOperator.ALL, value=["medium", "high"]
                )
            ]
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertEqual(len(result.ids), 2)

    def test_query_with_is_empty_filter_returns_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="not_existed_key", operator=FilterOperator.IS_EMPTY, value=None
                )
            ]
        )
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = simple_vector_store.query(query)
        self.assertEqual(len(result.ids), len(_node_embeddings_for_test()))

    def test_clear(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())
        simple_vector_store.clear()
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], similarity_top_k=3)
        result = simple_vector_store.query(query)
        self.assertEqual(len(result.ids), 0)

    def test_delete_nodes(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())
        simple_vector_store.delete_nodes(
            [_NODE_ID_WEIGHT_1_RANK_A, _NODE_ID_WEIGHT_2_RANK_C]
        )
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], similarity_top_k=3)
        result = simple_vector_store.query(query)
        self.assertEqual(result.ids, [_NODE_ID_WEIGHT_3_RANK_C])


def test_from_persist_dir(persist_dir: str) -> None:
    vector_store = SimpleVectorStore.from_persist_dir(persist_dir=persist_dir)
    assert vector_store is not None


def test_from_namespaced_persist_dir(persist_dir: str) -> None:
    vector_store = SimpleVectorStore.from_namespaced_persist_dir(
        persist_dir=persist_dir
    )
    assert vector_store is not None
