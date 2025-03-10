from llama_index.postprocessor.contextual_rerank import ContextualRerank
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from contextual.types import RerankCreateResponse
from unittest import mock, TestCase


class TestContextualRerank(TestCase):
    def test_contextual_rerank(self):
        nodes = [
            NodeWithScore(node=TextNode(text="the capital of france is paris")),
            NodeWithScore(
                node=TextNode(text="the capital of the United States is Washington DC")
            ),
        ]

        exp_rerank_response = RerankCreateResponse(
            results=[
                {"index": 0, "relevance_score": 0.616},
                {"index": 1, "relevance_score": 0.445},
            ]
        )

        expected_nodes = [
            NodeWithScore(
                node=TextNode(text="the capital of france is paris"), score=0.616
            ),
            NodeWithScore(
                node=TextNode(text="the capital of the United States is Washington DC"),
                score=0.445,
            ),
        ]

        query = "What is the capital of France?"

        # Mock the ContextualAI client
        contextual_client = mock.MagicMock()
        contextual_client.rerank.create.return_value = exp_rerank_response

        contextual_rerank = ContextualRerank(
            api_key="blah",
            model="ctxl-rerank-en-v1-instruct",
            top_n=2,
            client=contextual_client,
        )

        actual_nodes = contextual_rerank.postprocess_nodes(nodes, query_str=query)
        assert len(actual_nodes) == 2
        for actual_node_with_score, expected_node_with_score in zip(
            actual_nodes, expected_nodes
        ):
            print(actual_node_with_score.score)
            self.assertEqual(
                actual_node_with_score.node.get_content(),
                expected_node_with_score.node.get_content(),
            )
            self.assertAlmostEqual(
                actual_node_with_score.score,
                expected_node_with_score.score,
                places=3,
            )

    def test_class(self):
        names_of_base_classes = [b.__name__ for b in ContextualRerank.__mro__]
        assert BaseNodePostprocessor.__name__ in names_of_base_classes
