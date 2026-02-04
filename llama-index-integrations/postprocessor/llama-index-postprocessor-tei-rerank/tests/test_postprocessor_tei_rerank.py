from unittest.mock import patch

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
