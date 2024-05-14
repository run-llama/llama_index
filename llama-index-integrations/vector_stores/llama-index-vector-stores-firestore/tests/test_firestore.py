import json
import logging
from typing import Any, List
from unittest import TestCase
from unittest.mock import Mock, patch

import pytest

from google.cloud.firestore import DocumentReference, DocumentSnapshot, FieldFilter
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import node_to_metadata_dict

from llama_index.vector_stores.firestore import FirestoreVectorStore
from llama_index.vector_stores.firestore.utils import USER_AGENT

TEST_COLLECTION = "mock_collection"
TEST_EMBEDDING = [1.0, 2.0, 3.0]


@pytest.fixture(autouse=True, name="mock_client")
def mock_firestore_client() -> Any:
    """Returns a mock Firestore client."""
    with patch("google.cloud.firestore.Client") as mock_client_cls:
        with patch("importlib.metadata.version", return_value="0.1.0", autospec=True):
            mock_client = mock_client_cls.return_value
            mock_client._client_info.user_agent = USER_AGENT
            yield mock_client


@pytest.fixture(autouse=True, name="test_case")
def init_test_case() -> TestCase:
    """Returns a TestCase instance."""
    return TestCase()


@pytest.fixture(name="docs")
def document_snapshots() -> List[DocumentSnapshot]:
    """Returns a list of DocumentSnapshot instances."""
    return [
        DocumentSnapshot(
            reference=DocumentReference(TEST_COLLECTION, "aaa"),
            data={
                "embedding": Vector(TEST_EMBEDDING),
                "metadata": {
                    "_node_content": json.dumps(
                        {
                            "id_": "aaa",
                            "embedding": None,
                            "metadata": {"test_key": "test_value"},
                            "excluded_embed_metadata_keys": [],
                            "excluded_llm_metadata_keys": [],
                            "relationships": {},
                            "text": "dolor sit amet",
                            "start_char_idx": None,
                            "end_char_idx": None,
                            "text_template": "{metadata_str}\n\n{content}",
                            "metadata_template": "{key}: {value}",
                            "metadata_seperator": "\n",
                            "class_name": "TextNode",
                        }
                    ),
                    "_node_type": "TextNode",
                    "doc_id": "aaa",
                    "document_id": "aaa",
                    "ref_doc_id": "1234",
                },
            },
            exists=True,
            read_time=None,
            create_time=None,
            update_time=None,
        ),
    ]


@pytest.fixture(name="vector_store")
def firestore_vector_store(
    mock_client: Mock, docs: List[DocumentSnapshot]
) -> FirestoreVectorStore:
    """Returns a FirestoreVectorStore instance."""
    with patch("importlib.metadata.version", return_value="0.1.0"):
        mock_collection = mock_client.collection.return_value
        mock_collection.find_nearest.return_value.get.return_value = docs
        return FirestoreVectorStore(mock_client, collection_name=TEST_COLLECTION)


def _get_sample_vector(num: float) -> List[float]:
    """
    Get sample embedding vector of the form [num, 1, 1, ..., 1]
    where the length of the vector is 10.
    """
    return [num] + [1.0] * 10


@pytest.mark.parametrize(
    "sample_nodes",
    [
        (
            [
                TextNode(
                    text="lorem ipsum",
                    id_="aaa",
                    embedding=_get_sample_vector(1.0),
                ),
                TextNode(
                    text="dolor sit amet",
                    id_="bbb",
                    extra_info={"test_key": "test_value"},
                    embedding=_get_sample_vector(0.1),
                ),
                TextNode(
                    text="The quick brown fox jumped over the lazy dog.",
                    id_="ccc",
                    embedding=_get_sample_vector(5.0),
                ),
            ]
        )
    ],
)
def test_add_vectors(
    vector_store: FirestoreVectorStore,
    sample_nodes: List[BaseNode],
    mock_client: Mock,
    test_case: TestCase,
) -> None:
    """Test adding TextNodes to Firestore."""
    result_ids = vector_store.add(sample_nodes)
    batch_mock = mock_client.batch.return_value

    test_case.assertListEqual(result_ids, ["aaa", "bbb", "ccc"])
    test_case.assertEqual(mock_client.batch.call_count, 1)
    test_case.assertEqual(batch_mock.set.call_count, len(sample_nodes))

    for i, node in enumerate(sample_nodes):
        expected_metadata = node_to_metadata_dict(
            node, remove_text=False, flat_metadata=True
        )

        expected_entry = {
            "embedding": Vector(node.get_embedding()),
            "metadata": expected_metadata,
        }

    call_args = batch_mock.set.call_args_list[i][0]

    _, entry = call_args

    assert entry == expected_entry


def test_delete_node(vector_store: FirestoreVectorStore, mock_client: Mock) -> None:
    """Test deleting a node from Firestore."""
    vector_store.delete("1234")

    mock_client.collection.assert_called_with("mock_collection")
    collection_mock = mock_client.collection.return_value
    collection_mock.where.assert_called_with("metadata.ref_doc_id", "==", "1234")


def test_query(
    vector_store: FirestoreVectorStore,
    docs: List[DocumentSnapshot],
    test_case: TestCase,
) -> None:
    """Test querying the vector store."""
    query_embedding = [1.0, 2.0, 3.0]
    query = VectorStoreQuery(query_embedding=query_embedding)
    result = vector_store.query(query)

    nodes_list = list(result.nodes)

    assert isinstance(result, VectorStoreQueryResult)
    assert len(nodes_list) == len(docs)
    assert len(result.ids or []) == len(docs)

    for node, expected_result in zip(nodes_list, docs):
        expected_data = TextNode.from_json(
            expected_result.get("metadata").get("_node_content")
        )
        assert isinstance(node, TextNode)
        assert node.id_ == expected_data.id_
        assert node.text == expected_data.text

    vector_store.client.collection.assert_called_with(TEST_COLLECTION)
    vector_store.client.collection.return_value.find_nearest.assert_called_with(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.COSINE,
        limit=1,
    )


logger = logging.getLogger(__name__)


def test_query_with_field_filter(
    vector_store: FirestoreVectorStore,
    test_case: TestCase,
) -> None:
    """Test querying the vector store with filters."""
    query_embedding = [1.0, 2.0, 3.0]
    query = VectorStoreQuery(
        query_embedding=query_embedding,
        filters=MetadataFilters(
            filters=[MetadataFilter(key="metadata.ref_doc_id", value="1234")]
        ),
    )
    collection_ref_with_filter = (
        vector_store.client.collection.return_value.where.return_value
    )

    vector_store.query(query)

    vector_store.client.collection.assert_called_with(TEST_COLLECTION)
    collection_ref_with_filter.find_nearest.assert_called_with(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.COSINE,
        limit=1,
    )
    test_case.assertIsInstance(
        vector_store.client.collection.return_value.where.call_args_list[0][1][
            "filter"
        ],
        FieldFilter,
    )
