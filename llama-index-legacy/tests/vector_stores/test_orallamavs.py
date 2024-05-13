# OopCompanion:suppressRename
import traceback
from typing import List, Optional

import oracledb
import torch
from dotenv import load_dotenv
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.legacy.vector_stores import OraLlamaVS, orallamavs
from llama_index.legacy.vector_stores.orallamavs import DistanceStrategy
from oracledb import Connection

_NODE_ID_WEIGHT_1_RANK_A = "AF3BE6C4-5F43-4D74-B075-6B0E07900DE8"
_NODE_ID_WEIGHT_2_RANK_C = "7D9CD555-846C-445C-A9DD-F8924A01411D"
_NODE_ID_WEIGHT_3_RANK_C = "452D24AB-F185-414C-A352-590B4B9EE51B"


def db_connect(
    host: str = "152.67.235.198",
    port: int = 1521,
    user: str = "vector",
    passwd: str = "vector",
    svc_name: str = "orclpdb1",
) -> Connection:
    conn: Optional[Connection] = None
    try:
        dsn = oracledb.makedsn(host, port, service_name=svc_name)
        conn = oracledb.connect(user=user, password=passwd, dsn=dsn)
    except Exception as ex:
        print("An exception occured ::", ex)
        traceback.print_exc()
    finally:
        conn = conn if conn is not None else None
        return conn


def _node_embeddings_for_test() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_=_NODE_ID_WEIGHT_1_RANK_A,
            embedding=[1.0, 0.0],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
            metadata={"weight": 1.0, "rank": "a"},
        ),
        TextNode(
            text="lorem ipsum",
            id_=_NODE_ID_WEIGHT_2_RANK_C,
            embedding=[0.0, 1.0],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={"weight": 2.0, "rank": "c"},
        ),
        TextNode(
            text="lorem ipsum",
            id_=_NODE_ID_WEIGHT_3_RANK_C,
            embedding=[1.0, 1.0],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
            metadata={"weight": 3.0, "rank": "c"},
        ),
    ]


class OraLlamaVSTest:
    def __init__(self):
        load_dotenv()
        device = "gpu" if torch.cuda.is_available() else "cpu"
        self.client = db_connect()
        self.table_name = "llama_index8"

        self.orallamavs = OraLlamaVS(
            self.client, self.table_name, DistanceStrategy.DOT_PRODUCT, 2
        )

        self.orallamavs.add(_node_embeddings_for_test())

    def test_create_index(self):
        try:
            orallamavs.create_index(
                self.client,
                self.orallamavs,
                params={"idx_name": "hnsw_idx1", "idx_type": "HNSW"},
            )

            # orallamavs.create_index(self.client, self.orallamavs, params={
            #   "idx_name": "ivf_idx3", "idx_type": "IVF", "neighbor_part": 64, "accuracy": 90
            # })
            print("Index created successfully")
            return
        except Exception as ex:
            print("Exception occurred while index creation", ex)
            traceback.print_exc()

    def test_query_without_filters_returns_all_rows_sorted_by_similarity(self) -> None:
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], similarity_top_k=3)
        result = self.orallamavs.query(query=query)
        assert result.ids is not None
        assert sorted(result.ids) == sorted(
            [
                _NODE_ID_WEIGHT_1_RANK_A,
                _NODE_ID_WEIGHT_2_RANK_C,
                _NODE_ID_WEIGHT_3_RANK_C,
            ]
        )
        print(result.ids[0])
        assert result.ids[0] == _NODE_ID_WEIGHT_3_RANK_C

    def test_query_with_filters_returns_multiple_matches(self) -> None:
        filters = MetadataFilters(filters=[ExactMatchFilter(key="rank", value="c")])
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
        )
        result = self.orallamavs.query(query)
        print(result.ids)
        assert sorted(result.ids) == sorted(
            [_NODE_ID_WEIGHT_3_RANK_C, _NODE_ID_WEIGHT_2_RANK_C]
        )

    def test_query_with_filter_applies_top_k(self) -> None:
        filters = MetadataFilters(filters=[ExactMatchFilter(key="rank", value="c")])
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=1
        )
        result = self.orallamavs.query(query)
        print(result.ids)
        assert result.ids == [_NODE_ID_WEIGHT_3_RANK_C]

    def test_query_with_filter_applies_node_id_filter(self) -> None:
        filters = MetadataFilters(filters=[ExactMatchFilter(key="rank", value="c")])
        query = VectorStoreQuery(
            query_embedding=[1.0, 1.0],
            filters=filters,
            similarity_top_k=3,
            node_ids=[_NODE_ID_WEIGHT_3_RANK_C],
        )
        result = self.orallamavs.query(query)
        print(result.ids)
        assert result.ids == [_NODE_ID_WEIGHT_3_RANK_C]

    def test_query_with_exact_filters_returns_single_match(self) -> None:
        filters = MetadataFilters(
            filters=[
                ExactMatchFilter(key="rank", value="c"),
                ExactMatchFilter(key="weight", value=2),
            ]
        )
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
        result = self.orallamavs.query(query)
        print(result.ids)
        assert result.ids == [_NODE_ID_WEIGHT_2_RANK_C]

    def test_query_with_contradictive_filter_returns_no_matches(self) -> None:
        filters = MetadataFilters(
            filters=[
                ExactMatchFilter(key="weight", value=2),
                ExactMatchFilter(key="weight", value=3),
            ]
        )
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
        result = self.orallamavs.query(query)
        print(result.ids)
        assert result.ids is not None
        assert len(result.ids) == 0

    def test_query_with_filter_on_unknown_field_returns_no_matches(self) -> None:
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="unknown_field", value="c")]
        )
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
        result = self.orallamavs.query(query)
        print(result.ids)
        assert result.ids is not None
        assert len(result.ids) == 0

    def test_delete_removes_document_from_query_results(self) -> None:
        self.orallamavs.delete("test-1")
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], similarity_top_k=2)
        result = self.orallamavs.query(query)
        print(result.ids)
        assert sorted(result.ids) == sorted(
            [_NODE_ID_WEIGHT_3_RANK_C, _NODE_ID_WEIGHT_1_RANK_A]
        )


if __name__ == "__main__":
    ovst = OraLlamaVSTest()
    ovst.test_create_index()
    # ovst.test_query_without_filters_returns_all_rows_sorted_by_similarity()
    ovst.test_query_with_filters_returns_multiple_matches()
    ovst.test_query_with_filter_applies_top_k()
    ovst.test_query_with_filter_applies_node_id_filter()
    ovst.test_query_with_exact_filters_returns_single_match()
    ovst.test_query_with_contradictive_filter_returns_no_matches()
    ovst.test_query_with_filter_on_unknown_field_returns_no_matches()
    ovst.test_delete_removes_document_from_query_results()

    exit(0)
