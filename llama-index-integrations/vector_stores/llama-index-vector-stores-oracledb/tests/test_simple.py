import os
import threading
import unittest
from typing import List

import oracledb
import pytest

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores import FilterCondition, FilterOperator
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.oracledb import OraLlamaVS
from llama_index.vector_stores.oracledb import base as orallamavs

username = os.environ.get("VECDB_USER")
password = os.environ.get("VECDB_PASS")
dsn = os.environ.get("VECDB_HOST")

connection = oracledb.create_pool(user=username, password=password, dsn=dsn, max=4)

_NODE_ID_WEIGHT_1_RANK_A = "AF3BE6C4-5F43-4D74-B075-6B0E07900DE8"
_NODE_ID_WEIGHT_2_RANK_C = "7D9CD555-846C-445C-A9DD-F8924A01411D"
_NODE_ID_WEIGHT_3_RANK_C = "452D24AB-F185-414C-A352-590B4B9EE51B"


@pytest.fixture(autouse=True)
def drop_table():
    orallamavs.drop_table_purge(connection, "TABLEHELLO")
    yield
    orallamavs.drop_table_purge(connection, "TABLEHELLO")


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


class OraLlamaVSTest(unittest.TestCase):
    def test_query_without_filters_returns_all_rows_sorted_by_similarity(self) -> None:
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(filters=[ExactMatchFilter(key="rank", value="c")])
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=1
        )
        result = simple_vector_store.query(query)
        self.assertEqual(result.ids, [_NODE_ID_WEIGHT_3_RANK_C])

    def test_query_with_filter_applies_node_id_filter(self) -> None:
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="unknown_field", value="c")]
        )
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
        result = simple_vector_store.query(query)
        assert result.ids is not None
        self.assertEqual(len(result.ids), 0)

    def test_delete_removes_document_from_query_results(self) -> None:
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
        simple_vector_store.add(_node_embeddings_for_test())

        simple_vector_store.delete("test-1")
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], similarity_top_k=2)
        result = simple_vector_store.query(query)
        self.assertEqual(
            result.ids,
            [_NODE_ID_WEIGHT_3_RANK_C, _NODE_ID_WEIGHT_1_RANK_A],
        )

    def test_query_with_filters_with_filter_condition(self) -> None:
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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

    def test_query_with_is_empty_filter_returns_matches(self) -> None:
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
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

    def test_thread(self) -> None:
        # 6. Add 2 different record concurrently
        # Expectation:Successful
        simple_vector_store = OraLlamaVS(connection, "TABLEHELLO")
        simple_vector_store.add(_node_embeddings_for_test())

        def add_sss(inc: int) -> None:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="not_existed_key",
                        operator=FilterOperator.IS_EMPTY,
                        value=None,
                    )
                ]
            )
            query = VectorStoreQuery(
                query_embedding=[1.0 + inc, 1.0], filters=filters, similarity_top_k=3
            )
            return simple_vector_store.query(query)

        threads = []
        for i in range(10):
            thread_1 = threading.Thread(target=add_sss, args=[i])
            threads.append(thread_1)

        for thread_ in threads:
            thread_.start()
        for thread_ in threads:
            thread_.join()

        assert connection.busy == 0
