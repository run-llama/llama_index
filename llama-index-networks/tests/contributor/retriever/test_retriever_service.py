from typing import List
from fastapi.testclient import TestClient
from llama_index.networks.contributor.retriever import (
    ContributorRetrieverService,
    ContributorRetrieverServiceSettings,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle


class MockRetriever(BaseRetriever):
    """Custom retriever for testing."""

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Mock retrieval."""
        return [
            NodeWithScore(
                node=TextNode(text=f"mock_{query_bundle.query_str}"), score=0.9
            )
        ]

    async def _aretrieve(self, query_bundle: str) -> List[NodeWithScore]:
        """Mock retrieval."""
        return [
            NodeWithScore(
                node=TextNode(text=f"mock_{query_bundle.query_str}"), score=0.9
            )
        ]


def test_contributor_service_index():
    # arrange
    config = ContributorRetrieverServiceSettings()
    mock_retriever = MockRetriever()
    service = ContributorRetrieverService(retriever=mock_retriever, config=config)
    test_client = TestClient(service.app)

    # act
    response = test_client.get("/api")

    # assert
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World!"}


def test_contributor_service_retrieve():
    # arrange
    config = ContributorRetrieverServiceSettings(secret="secret")
    mock_retriever = MockRetriever()
    service = ContributorRetrieverService(retriever=mock_retriever, config=config)
    test_client = TestClient(service.app)

    # act
    response = test_client.post("/api/retrieve", json={"query": "mock_query"})
    nodes_dict = response.json()["nodes_dict"]

    # assert
    assert response.status_code == 200
    assert len(nodes_dict) == 1
    assert nodes_dict[0]["node"]["text"] == "mock_mock_query"
    assert nodes_dict[0]["score"] == 0.9
