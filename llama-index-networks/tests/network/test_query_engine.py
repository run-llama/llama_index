from llama_index.networks.network.query_engine import NetworkQueryEngine
from llama_index.core.llms.mock import MockLLM
from llama_index.core.base.response.schema import Response
from unittest.mock import patch


async def return_response():
    return Response(response="mock contributor response", metadata={"score": 0.5})


@patch(
    "llama_index.networks.contributor.query_engine.client.ContributorQueryEngineClient"
)
def test_network_query_engine(mock_contributor):
    # arange
    llm = MockLLM()
    mock_contributor.aquery.return_value = return_response()
    network_query_engine = NetworkQueryEngine.from_args(
        contributors=[mock_contributor], llm=llm
    )

    # act
    _ = network_query_engine.query("Are you a mock?")

    # assert
    mock_contributor.aquery.assert_called_once()
    args, _ = mock_contributor.aquery.call_args
    query_bundle = args[0]
    assert query_bundle.query_str == "Are you a mock?"
