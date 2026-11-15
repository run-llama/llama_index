import json
from unittest.mock import patch

import httpx
import pytest

from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.postprocessor.tei_rerank import TextEmbeddingInference


def test_class():
    names_of_base_classes = [b.__name__ for b in TextEmbeddingInference.__mro__]

    assert BaseNodePostprocessor.__name__ in names_of_base_classes


def test_rerank_logic():
    nodes = [
        NodeWithScore(node=TextNode(text="text1"), score=0.5),
        NodeWithScore(node=TextNode(text="text2"), score=0.5),
    ]
    query_bundle = QueryBundle(query_str="test query")

    # Mock the _call_api method to return scores with indices
    with patch.object(
        TextEmbeddingInference,
        "_call_api",
        return_value=[
            {"index": 1, "score": 0.9},
            {"index": 0, "score": 0.1},
        ],
    ) as mock_call_api:
        # Test Case 1: Basic Reranking
        postprocessor = TextEmbeddingInference(top_n=2)
        new_nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)

        # Verify API call
        mock_call_api.assert_called_once()

        # Check if nodes are sorted by score
        assert len(new_nodes) == 2
        assert new_nodes[0].node.text == "text2"
        assert new_nodes[0].score == 0.9
        assert new_nodes[1].node.text == "text1"
        assert new_nodes[1].score == 0.1

    # Test Case 2: Keep Retrieval Score
    with patch.object(
        TextEmbeddingInference,
        "_call_api",
        return_value=[
            {"index": 1, "score": 0.9},
            {"index": 0, "score": 0.1},
        ],
    ):
        postprocessor = TextEmbeddingInference(top_n=2)
        postprocessor.keep_retrieval_score = True

        # Reset scores to ensure we are testing the preservation
        nodes[0].score = 0.5
        nodes[1].score = 0.5

        new_nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)

        assert new_nodes[0].node.metadata["retrieval_score"] == 0.5
        assert new_nodes[1].node.metadata["retrieval_score"] == 0.5


def _mock_client(handler):
    real_client = httpx.Client
    return lambda *args, **kwargs: real_client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("truncate_text", [True, False])
def test_call_api_sends_truncate(truncate_text):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=[{"index": 0, "score": 0.9}])

    kwargs = {} if truncate_text else {"truncate_text": False}
    postprocessor = TextEmbeddingInference(**kwargs)
    with patch("httpx.Client", _mock_client(handler)):
        postprocessor._call_api("test query", ["text1"])

    assert captured["truncate"] is truncate_text


def test_call_api_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            413,
            json={
                "error": "Input validation error: `inputs` must have less than 512 tokens. Given: 1024",
                "error_type": "Validation",
            },
        )

    nodes = [NodeWithScore(node=TextNode(text=f"text{i}")) for i in range(3)]
    postprocessor = TextEmbeddingInference()
    with patch("httpx.Client", _mock_client(handler)):
        with pytest.raises(httpx.HTTPStatusError):
            postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(query_str="test query")
            )
