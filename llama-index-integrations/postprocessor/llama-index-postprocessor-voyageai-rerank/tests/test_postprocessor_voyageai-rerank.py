from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from voyageai.api_resources import VoyageResponse

from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank
from voyageai.object.reranking import RerankingObject

rerank_sample_response = {
    "object": "list",
    "data": [
        {"relevance_score": 0.8984375, "index": 1},
        {"relevance_score": 0.5234375, "index": 0},
    ],
    "model": "rerank-lite-1",
    "usage": {"total_tokens": 0},
}


def test_class():
    names_of_base_classes = [b.__name__ for b in VoyageAIRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


def test_rerank(mocker):
    # Mocked client with the desired behavior for embed_documents
    mock_client = mocker.MagicMock()
    mock_client.rerank.return_value = RerankingObject(
        documents=["0", "1"],
        response=VoyageResponse.construct_from(rerank_sample_response),
    )

    # Mock create_client to return our mock_client
    mocker.patch.object(VoyageAIRerank, "_client", return_value=mock_client)

    voyageai_rerank = VoyageAIRerank(
        api_key="api_key", top_n=2, model="rerank-lite-1", truncation=True
    )
    voyageai_rerank._postprocess_nodes(
        nodes=[NodeWithScore(node=TextNode(text="text"))],
        query_bundle=QueryBundle(query_str="any query"),
    )
