from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from voyageai.api_resources import VoyageResponse

from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank
from voyageai.object.reranking import RerankingObject
from pytest_mock import MockerFixture

rerank_sample_response = {
    "object": "list",
    "data": [
        {"relevance_score": 0.8984375, "index": 1},
        {"relevance_score": 0.5234375, "index": 0},
    ],
    "model": "rerank-lite-1",
    "usage": {"total_tokens": 100},
}


def test_class():
    names_of_base_classes = [b.__name__ for b in VoyageAIRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


def test_rerank(mocker: MockerFixture) -> None:
    # Mocked client with the desired behavior for embed_documents
    result_object = RerankingObject(
        documents=["0", "1"],
        response=VoyageResponse.construct_from(rerank_sample_response),
    )
    mock_client = mocker.MagicMock()
    mock_client.rerank.return_value = result_object

    # Mock create_client to return our mock_client
    mocker.patch(
        "llama_index.postprocessor.voyageai_rerank.VoyageAIRerank._client",
        return_value=mock_client,
        new_callable=mocker.PropertyMock,
    )

    voyageai_rerank = VoyageAIRerank(
        api_key="api_key", top_n=2, model="rerank-lite-1", truncation=True
    )
    result = voyageai_rerank.postprocess_nodes(
        nodes=[
            NodeWithScore(node=TextNode(text="text1")),
            NodeWithScore(node=TextNode(text="text2")),
        ],
        query_bundle=QueryBundle(query_str="any query"),
    )
    assert len(result) == 2
    assert result[0].text == "text2"
    assert result[1].text == "text1"
