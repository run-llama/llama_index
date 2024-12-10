from unittest import TestCase, mock

import boto3
from llama_index.core.postprocessor.types import (
    BaseNodePostprocessor,
    NodeWithScore,
    QueryBundle,
)
from llama_index.core.schema import TextNode

from llama_index.postprocessor.bedrock_rerank import AWSBedrockRerank


class TestAWSBedrockRerank(TestCase):
    def test_class(self):
        names_of_base_classes = [b.__name__ for b in AWSBedrockRerank.__mro__]
        self.assertIn(BaseNodePostprocessor.__name__, names_of_base_classes)

    def test_bedrock_rerank(self):
        exp_rerank_response = {
            "results": [
                {
                    "index": 2,
                    "relevanceScore": 0.9,
                },
                {
                    "index": 3,
                    "relevanceScore": 0.8,
                },
            ]
        }

        input_nodes = [
            NodeWithScore(node=TextNode(id_="1", text="first 1")),
            NodeWithScore(node=TextNode(id_="2", text="first 2")),
            NodeWithScore(node=TextNode(id_="3", text="last 1")),
            NodeWithScore(node=TextNode(id_="4", text="last 2")),
        ]

        expected_nodes = [
            NodeWithScore(node=TextNode(id_="3", text="last 1"), score=0.9),
            NodeWithScore(node=TextNode(id_="4", text="last 2"), score=0.8),
        ]

        bedrock_client = boto3.client("bedrock-agent-runtime", region_name="us-west-2")
        reranker = AWSBedrockRerank(client=bedrock_client, num_results=2)

        with mock.patch.object(
            bedrock_client, "rerank", return_value=exp_rerank_response
        ):
            query_bundle = QueryBundle(query_str="last")

            actual_nodes = reranker.postprocess_nodes(
                input_nodes, query_bundle=query_bundle
            )

            self.assertEqual(len(actual_nodes), len(expected_nodes))
            for actual_node_with_score, expected_node_with_score in zip(
                actual_nodes, expected_nodes
            ):
                self.assertEqual(
                    actual_node_with_score.node.get_content(),
                    expected_node_with_score.node.get_content(),
                )
                self.assertAlmostEqual(
                    actual_node_with_score.score, expected_node_with_score.score
                )
