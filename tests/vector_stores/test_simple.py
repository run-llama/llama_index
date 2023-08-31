from typing import List
import unittest

from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores import SimpleVectorStore
from llama_index.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    NodeWithEmbedding,
    VectorStoreQuery,
)

_NODE_ID_WEIGHT_1_RANK_A = "AF3BE6C4-5F43-4D74-B075-6B0E07900DE8"
_NODE_ID_WEIGHT_2_RANK_C = "7D9CD555-846C-445C-A9DD-F8924A01411D"
_NODE_ID_WEIGHT_3_RANK_C = "452D24AB-F185-414C-A352-590B4B9EE51B"


def _node_embeddings_for_test() -> List[NodeWithEmbedding]:
    return [
        NodeWithEmbedding(
            embedding=[1.0, 0.0],
            node=TextNode(
                text="lorem ipsum",
                id_=_NODE_ID_WEIGHT_1_RANK_A,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")
                },
                metadata={"weight": 1.0, "rank": "a"},
            ),
        ),
        NodeWithEmbedding(
            embedding=[0.0, 1.0],
            node=TextNode(
                text="lorem ipsum",
                id_=_NODE_ID_WEIGHT_2_RANK_C,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")
                },
                metadata={"weight": 2.0, "rank": "c"},
            ),
        ),
        NodeWithEmbedding(
            embedding=[1.0, 1.0],
            node=TextNode(
                text="lorem ipsum",
                id_=_NODE_ID_WEIGHT_3_RANK_C,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")
                },
                metadata={"weight": 3.0, "rank": "c"},
            ),
        ),
    ]


class SimpleVectorStoreTest(unittest.TestCase):
    def test_query_without_filters_returns_all_rows_sorted_by_similarity(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        query = VectorStoreQuery(query_embedding=[1.0, 1.0], similarity_top_k=3)
        result = simple_vector_store.query(query)
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
        self.assertEqual(len(result.ids), 0)

    def test_query_with_filter_on_unknown_field_returns_no_matches(self) -> None:
        simple_vector_store = SimpleVectorStore()
        simple_vector_store.add(_node_embeddings_for_test())

        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="unknown_field", value="c")]
        )
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
        result = simple_vector_store.query(query)
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
