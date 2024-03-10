from llama_index.networks.network.retriever import NetworkRetriever
from llama_index.core.base.response.schema import NodeWithScore
from unittest.mock import patch


async def return_nodes():
    return [NodeWithScore(node={"id_": "mock_node", "metadata": {}}, score=0.9)]


@patch("llama_index.networks.contributor.retriever.client.ContributorClient")
def test_network_retriever(mock_contributor):
    # arrange
    mock_contributor.aretrieve.return_value = return_nodes()
    network_retriever = NetworkRetriever(contributors=[mock_contributor])

    # act
    result = network_retriever.retrieve("What's the scenario?")

    # assert
    mock_contributor.aretrieve.assert_called_once()
    args, _ = mock_contributor.aretrieve.call_args
    query_str = args[0]
    assert query_str == "What's the scenario?"
    assert len(result) == 1
    assert result[0].node["id_"] == "mock_node"
    assert result[0].score == 0.9
