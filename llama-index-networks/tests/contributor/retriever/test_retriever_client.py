from unittest.mock import patch, MagicMock
from llama_index.networks.contributor.retriever import (
    ContributorRetrieverClient,
    ContributorRetrieverClientSettings,
)
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.settings import Settings
from llama_index.core.schema import QueryBundle


@patch("llama_index.networks.contributor.retriever.client.requests.post")
def test_contributor_client_retrieve(mock_post):
    # Arrange
    mock_node_1 = NodeWithScore(node=TextNode(id_="node1", text="mock 1"), score=0.9)
    mock_node_2 = NodeWithScore(node=TextNode(id_="node2", text="mock 2"), score=0.8)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "nodes_dict": [
            mock_node_1.to_dict(),
            mock_node_2.to_dict(),
        ]
    }
    mock_post.return_value = mock_response

    settings = ContributorRetrieverClientSettings(api_url="fake-url")
    client = ContributorRetrieverClient(
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
    assert nodes[0].node.id_ == "node1"
    assert nodes[1].score == 0.8
