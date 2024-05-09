import pytest

from unittest.mock import MagicMock, patch, AsyncMock

from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult

from manager_client import (
    PageVectorSearchQueryResponseItem,
    VectorSearchQueryResponseItem,
)
from llama_index.vector_stores.wordlift import WordliftVectorStore


DUMMY_KEY = "dummy_key"


class MockNode:
    def __init__(self, node_id, embedding, content):
        self.node_id = node_id
        self._embedding = embedding
        self._content = content

    def dict(self):
        return {"metadata": {}, "entity_id": "123", "node_id": self.node_id}

    def get_embedding(self):
        return self._embedding

    def get_content(self, metadata_mode):
        return self._content


@pytest.fixture()
def mock_vector_search_service():
    mock_vector_search_service = MagicMock()
    mock_vector_search_service.update_nodes_collection = AsyncMock(
        side_effect=async_update_nodes_collection
    )
    mock_vector_search_service.query_nodes_collection = AsyncMock(
        side_effect=async_query_nodes_collection
    )
    return mock_vector_search_service


@pytest.fixture()
def mock_key_provider():
    mock_key_provider = MagicMock()
    # Mock the key provider behavior
    mock_key_provider.for_add = AsyncMock(side_effect=async_mock_for_getting_key)
    mock_key_provider.for_query = AsyncMock(side_effect=async_mock_for_getting_key)
    return mock_key_provider


@pytest.fixture()
def wordlift_vector_store(mock_key_provider, mock_vector_search_service):
    return WordliftVectorStore(mock_key_provider, mock_vector_search_service)


async def async_mock_for_getting_key(nodes):
    return DUMMY_KEY


async def async_update_nodes_collection(node_request, key):
    return None


async def async_query_nodes_collection(vector_search_query_request, key):
    item_1 = VectorSearchQueryResponseItem(
        text="test item 1",
        node_id="123",
        embeddings=[0.1, 0.2, 0.3],
        metadata={},
        score=1,
    )
    item_2 = VectorSearchQueryResponseItem(
        text="test item 2",
        node_id="456",
        embeddings=[0.4, 0.5, 0.6],
        metadata={},
        score=1,
    )
    page = PageVectorSearchQueryResponseItem(
        first=None,
        items=[item_1, item_2],
        last=None,
        next=None,
        prev=None,
        var_self=None,
    )
    return page


def test_instance_creation_with_key_provider_service_and_vector_search_service(
    mock_key_provider, mock_vector_search_service
) -> None:
    store = WordliftVectorStore(mock_key_provider, mock_vector_search_service)
    assert isinstance(store, WordliftVectorStore)


def test_instance_creation_with_create_method() -> None:
    store = WordliftVectorStore.create(DUMMY_KEY)
    assert isinstance(store, WordliftVectorStore)


def test_add(
    wordlift_vector_store,
):
    # Create mock node data
    mock_nodes = [
        MockNode(node_id="1", embedding=[0.1, 0.2, 0.3], content="content 1"),
        MockNode(node_id="2", embedding=[0.4, 0.5, 0.6], content="content 2"),
    ]

    # Mock NodeRequest class
    with patch(
        "llama_index.vector_stores.wordlift.base.NodeRequest"
    ) as MockNodeRequest:
        # Call the add method
        result = wordlift_vector_store.add(mock_nodes)
        assert MockNodeRequest.call_count == 2
        wordlift_vector_store.vector_search_service.update_nodes_collection.assert_called_once()
        # Assert the behavior
        assert result == ["1", "2"]


@pytest.mark.xfail(raises=NotImplementedError)
def test_delete(wordlift_vector_store):
    wordlift_vector_store.delete("dummy_id")


def test_query(wordlift_vector_store):
    # Call the query method
    result = wordlift_vector_store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
    )
    wordlift_vector_store.vector_search_service.query_nodes_collection.assert_called_once()
    assert isinstance(result, VectorStoreQueryResult)
    assert result.nodes[0].get_content() == "test item 1"
    assert result.nodes[1].get_content() == "test item 2"
    assert result.ids == ["123", "456"]
