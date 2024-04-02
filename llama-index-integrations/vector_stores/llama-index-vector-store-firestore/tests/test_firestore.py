import json
from unittest.mock import patch
import pytest
from typing import List
from llama_index.core.schema import TextNode
from llama_index.vector_store.firestore import FirestoreVectorStore
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from google.cloud.firestore import (
    DocumentReference,
    DocumentSnapshot,
)

TEST_COLLECTION = "mock_collection"
TEST_EMBEDDING = [1.0, 2.0, 3.0]


@pytest.fixture(scope="module", autouse=True, name="mock_client")
def mock_firestore_client():
    """Returns a mock Firestore client."""
    with patch("google.cloud.firestore.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        yield mock_client


@pytest.fixture
def document_snapshots():
    """Returns a dict of mocked DocumentSnapshots."""
    return [
        DocumentSnapshot(
            reference=DocumentReference(TEST_COLLECTION, "aaa"),
            data={
                "embedding": TEST_EMBEDDING,
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
                },
            },
            exists=True,
            read_time=None,
            create_time=None,
            update_time=None,
        ),
    ]


@pytest.fixture
def firestore_vector_store(mock_client, document_snapshots):
    """Returns a FirestoreVectorStore instance."""
    mock_collection = mock_client.collection.return_value
    mock_collection.find_nearest.return_value.get.return_value = document_snapshots
    return FirestoreVectorStore(mock_client, collection_name=TEST_COLLECTION)


def _get_sample_vector(num: float) -> List[float]:
    """
    Get sample embedding vector of the form [num, 1, 1, ..., 1]
    where the length of the vector is 10.
    """
    return [num] + [1.0] * 10


@pytest.fixture
def sample_nodes():
    """Returns a list of sample TextNode instances."""
    return [
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
            index_id="ccc",
            embedding=_get_sample_vector(5.0),
        ),
    ]


def test_add_vectors(firestore_vector_store, sample_nodes, mock_client):
    """Test adding TextNodes to Firestore."""

    result_ids = firestore_vector_store.add(sample_nodes)

    assert result_ids == ["aaa", "bbb", "ccc"]
    mock_client.batch.assert_called()
    batch_mock = mock_client.batch.return_value
    batch_mock.commit.assert_called()

    for i, node in enumerate(sample_nodes):
        expected_metadata = node_to_metadata_dict(
            node, remove_text=False, flat_metadata=True
        )

        expected_entry = {
            "embedding": node.get_embedding(),
            "metadata": expected_metadata,
        }

        call_args = batch_mock.set.call_args_list[i][0]
        _, entry = call_args

        assert entry == expected_entry


def test_delete_node(firestore_vector_store, mock_client):
    """Test deleting a node from Firestore."""

    firestore_vector_store.delete("ref_doc_id")

    mock_client.collection.assert_called_with("mock_collection")
    collection_mock = mock_client.collection.return_value
    collection_mock.document.assert_called_with("ref_doc_id")
    document_mock = collection_mock.document.return_value
    document_mock.delete.assert_called()


def test_query(firestore_vector_store, document_snapshots):
    query_embedding = [1.0, 2.0, 3.0]
    query = VectorStoreQuery(query_embedding=query_embedding)
    result = firestore_vector_store.query(query)

    assert isinstance(result, VectorStoreQueryResult)
    assert len(result.nodes) == len(document_snapshots)
    assert len(result.ids) == len(document_snapshots)

    for node, expected_result in zip(result.nodes, document_snapshots):
        expected_data = TextNode.from_json(
            expected_result.get("metadata").get("_node_content")
        )
        assert isinstance(node, TextNode)
        assert node.id_ == expected_data.id_
        assert node.text == expected_data.text

    firestore_vector_store._client.collection.assert_called_with(TEST_COLLECTION)
    firestore_vector_store._client.collection.return_value.find_nearest.assert_called_with(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.COSINE,
        limit=10,
    )
