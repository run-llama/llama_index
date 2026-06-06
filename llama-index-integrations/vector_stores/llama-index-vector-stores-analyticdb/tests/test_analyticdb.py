from typing import List
from unittest.mock import MagicMock
import pytest

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter
from llama_index.vector_stores.analyticdb import AnalyticDBVectorStore
from llama_index.vector_stores.analyticdb.base import _build_filter_clause

try:
    from alibabacloud_gpdb20160503.client import Client

    analyticdb_installed = True
except ImportError:
    analyticdb_installed = False


def _create_mock_vector_store(client: Client):
    return AnalyticDBVectorStore(
        client=client,
        region_id="cn-hangzhou",
        instance_id="adb-instance-id",
        account="adb-account",
        account_password="adb-account-password",
    )


def _create_sample_documents(n: int) -> List[TextNode]:
    return [
        TextNode(
            text=f"text {i}",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id=f"test doc id {i}")
            },
            embedding=[0.5, 0.5, 0.5, 0.5],
        )
        for i in range(n)
    ]


@pytest.mark.skipif(
    not analyticdb_installed, reason="alibaba-adb-client package not installed"
)
def test_adbvector_add() -> None:
    client = MagicMock(spec=Client)
    vector_store = _create_mock_vector_store(client)

    nodes = _create_sample_documents(2)
    ids = vector_store.add(nodes)

    assert len(ids) == 2
    assert client.upsert_collection_data.call_count == 1


@pytest.mark.skipif(
    not analyticdb_installed, reason="alibaba-adb-client package not installed"
)
def test_adbvector_delete() -> None:
    client = MagicMock(spec=Client)
    vector_store = _create_mock_vector_store(client)

    vector_store.delete(ref_doc_id="31d00b73-41ca-4d5a-a5b4-85a014b2c924")

    assert client.delete_collection_data.call_count == 1


def test_build_filter_clause_escapes_sql_values() -> None:
    filter_clause = _build_filter_clause(
        MetadataFilter(
            key="author",
            value="x' OR '1'='1",
            operator=FilterOperator.EQ,
        )
    )

    assert filter_clause == "metadata_->>'author' = 'x'' OR ''1''=''1'"


def test_build_filter_clause_escapes_metadata_keys_and_values() -> None:
    assert (
        _build_filter_clause(
            MetadataFilter(
                key="source.file",
                value="O'Reilly",
                operator=FilterOperator.EQ,
            )
        )
        == "metadata_->>'source.file' = 'O''Reilly'"
    )

    assert (
        _build_filter_clause(
            MetadataFilter(
                key="content type",
                value="text/plain",
                operator=FilterOperator.EQ,
            )
        )
        == "metadata_->>'content type' = 'text/plain'"
    )

    assert (
        _build_filter_clause(
            MetadataFilter(
                key="author' OR '1'='1",
                value="x' OR '1'='1",
                operator=FilterOperator.EQ,
            )
        )
        == "metadata_->>'author'' OR ''1''=''1' = 'x'' OR ''1''=''1'"
    )
