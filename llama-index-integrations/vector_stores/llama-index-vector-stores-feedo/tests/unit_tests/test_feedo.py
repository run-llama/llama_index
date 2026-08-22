from typing import Any
import pytest
from pytest_mock import MockerFixture

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.feedo.base import FeedoVectorStore


@pytest.fixture()
def feedo_store(mocker: MockerFixture) -> FeedoVectorStore:
    # Mock the SearchModule import to avoid needing the real feedo-sdk installed
    mock_search_module = mocker.MagicMock()
    mocker.patch(
        "llama_index.vector_stores.feedo.base.SearchModule",
        return_value=mock_search_module,
        create=True
    )
    mocker.patch(
        "llama_index.vector_stores.feedo.base.NodeRouter",
        create=True
    )
    
    import sys
    sys.modules["feedo.router"] = mocker.MagicMock()
    sys.modules["feedo.modules.search"] = mocker.MagicMock()
    
    return FeedoVectorStore(usage_key="test_key", did="test_did", namespace="test_room")


def test_add_documents(feedo_store: FeedoVectorStore, mocker: MockerFixture) -> None:
    # Mock asyncio loop
    mock_loop = mocker.MagicMock()
    mocker.patch("asyncio.get_event_loop", return_value=mock_loop)
    
    nodes = [
        TextNode(text="Test content 1", id_="node1"),
        TextNode(text="Test content 2", id_="node2"),
    ]
    
    # Run add
    ids = feedo_store.add(nodes)
    
    # Verify
    assert len(ids) == 2
    assert ids == ["node1", "node2"]
    assert mock_loop.run_until_complete.call_count == 2


def test_query_documents(feedo_store: FeedoVectorStore, mocker: MockerFixture) -> None:
    mock_loop = mocker.MagicMock()
    mocker.patch("asyncio.get_event_loop", return_value=mock_loop)
    
    # Mock the return value of client.search
    mock_loop.run_until_complete.return_value = {
        "documents": [
            {"hash_id": "node1", "text": "Test content 1", "score": 0.95},
            {"hash_id": "node2", "text": "Test content 2", "score": 0.85},
        ]
    }
    
    query = VectorStoreQuery(query_str="test", similarity_top_k=2)
    result = feedo_store.query(query)
    
    assert len(result.nodes) == 2
    assert result.nodes[0].get_content() == "Test content 1"
    assert result.similarities[0] == 0.95
    assert result.ids[0] == "node1"
    
    assert result.nodes[1].get_content() == "Test content 2"
    assert result.similarities[1] == 0.85
    assert result.ids[1] == "node2"
