from typing import List
from fastapi.testclient import TestClient
from llama_index.networks.contributor.retriever import (
    ContributorService,
    ContributorServiceSettings,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, BaseNode


class MockRetriever(BaseRetriever):
    """Custom retriever for testing."""

    def retrieve(self, query_str: str) -> List[NodeWithScore]:
        """Mock retrieval."""
        return [NodeWithScore(node=BaseNode(id_=f"mock_{query_str}"), score=0.9)]

    async def aretrieve(self, query_str: str) -> List[NodeWithScore]:
        """Mock retrieval."""
        return [NodeWithScore(node=BaseNode(id_=f"mock_{query_str}"), score=0.9)]


def test_contributor_service_index():
    # arrange
    config = ContributorServiceSettings()
    mock_retriever = MockRetriever()
    service = ContributorService(retriever=mock_retriever, config=config)
    test_client = TestClient(service.app)

    # act
    response = test_client.get("/api")

    # assert
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World!"}


def test_contributor_service_retrieve():
    # arrange
    config = ContributorServiceSettings(secret="secret")
    mock_retriever = MockRetriever()
    service = ContributorService(retriever=mock_retriever, config=config)
    test_client = TestClient(service.app)

    # act
    response = test_client.post("/api/retrieve", json={"query": "mock_query"})

    # assert
    assert response.status_code == 200
    assert response.json() == [
        {"node": {"id_": "mock_mock_query", "metadata": None}, "score": 0.9}
    ]
