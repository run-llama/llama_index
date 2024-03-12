from llama_index.networks.network.retriever import NetworkRetriever
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from unittest.mock import patch


async def return_nodes():
    return [NodeWithScore(node=TextNode(text="mock_node"), score=0.9)]


@patch("llama_index.networks.contributor.retriever.client.ContributorRetrieverClient")
def test_network_retriever(mock_contributor):
    # arrange
    mock_contributor.aretrieve.return_value = return_nodes()
    network_retriever = NetworkRetriever(contributors=[mock_contributor])

    # act
    query_bundle = QueryBundle(query_str="What's the scenario?")
    result = network_retriever.retrieve(query_bundle)

    # assert
    mock_contributor.aretrieve.assert_called_once()
    args, _ = mock_contributor.aretrieve.call_args
    query_bundle = args[0]
    assert query_bundle.query_str == "What's the scenario?"
    assert len(result) == 1
    assert result[0].node.text == "mock_node"
    assert result[0].score == 0.9
