from unittest.mock import patch, MagicMock
from llama_index.networks.contributor.retriever import (
    ContributorClient,
    ContributorClientSettings,
)
from llama_index.core.settings import Settings
from llama_index.core.schema import QueryBundle


@patch("llama_index.networks.contributor.retriever.client.requests.post")
def test_contributor_client_retrieve(mock_post):
    # Arrange
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "nodes": [
            {"node": {"id_": "node1", "metadata": None}, "score": 0.9},
            {"node": {"id_": "node2", "metadata": None}, "score": 0.8},
        ]
    }
    mock_post.return_value = mock_response

    settings = ContributorClientSettings(api_url="fake-url")
    client = ContributorClient(
        config=settings, callback_manager=Settings.callback_manager
    )

    query_bundle = QueryBundle(query_str="Does this work?")

    # Act
    nodes = client.retrieve(query_bundle)

    # Assert
    mock_post.assert_called_once_with(
        "fake-url/api/retrieve",
        json={"query": "Does this work?", "api_key": None},
        headers={},
    )

    assert len(nodes) == 2
    assert nodes[0]["node"]["id_"] == "node1"
    assert nodes[1]["score"] == 0.8
