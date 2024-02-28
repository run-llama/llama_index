from fastapi.testclient import TestClient
from llama_index.networks import (
    ContributorService,
    ContributorServiceSettings,
)
from llama_index.core.query_engine.custom import CustomQueryEngine


class MockQueryEngine(CustomQueryEngine):
    """Custom query engine."""

    def custom_query(self, query_str: str) -> str:
        """Query."""
        return "custom_" + query_str

    async def acustom_query(self, query_str: str) -> str:
        """Query."""
        return "custom_" + query_str


def test_contributor_service_index():
    # arrange
    config = ContributorServiceSettings()
    mock_query_engine = MockQueryEngine()
    service = ContributorService(query_engine=mock_query_engine, config=config)
    test_client = TestClient(service.app)

    # act
    response = test_client.get("/api")

    # assert
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World!"}


def test_contributor_service_query():
    # arrange
    config = ContributorServiceSettings(secret="secret")
    mock_query_engine = MockQueryEngine()
    service = ContributorService(query_engine=mock_query_engine, config=config)
    test_client = TestClient(service.app)

    # act
    response = test_client.post("/api/query", json={"query": "What's up?"})

    # assert
    assert response.status_code == 200
    assert response.json() == {
        "response": "custom_What's up?",
        "source_nodes": [],
        "metadata": None,
    }
