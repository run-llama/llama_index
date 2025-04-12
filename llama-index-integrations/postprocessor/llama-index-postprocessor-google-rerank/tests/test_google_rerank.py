import os
import json
from unittest import TestCase, mock

from google.oauth2 import service_account
from google.cloud import discoveryengine_v1 as discoveryengine

from llama_index.core.postprocessor.types import (
    BaseNodePostprocessor,
    NodeWithScore,
    QueryBundle,
)
from llama_index.core.schema import TextNode
from llama_index.postprocessor.google_rerank import GoogleRerank


class TestAWSBedrockRerank(TestCase):
    def test_class(self):
        names_of_base_classes = [b.__name__ for b in GoogleRerank.__mro__]
        self.assertIn(BaseNodePostprocessor.__name__, names_of_base_classes)

    def test_bedrock_rerank(self):
        exp_rerank_response = {
            "records": [
                {
                    "id": "2",
                    "score": 0.9,
                },
                {
                    "id": "3",
                    "score": 0.8,
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

        gcp_param = json.loads(os.getenv("GOOGLE_CLOUD_CREDENTIALS", None))
        google_credentials = service_account.Credentials.from_service_account_info(gcp_param)
        reranker_client = discoveryengine.RankServiceClient(credentials=google_credentials)
        reranker = GoogleRerank(client=reranker_client, num_results=2)

        with mock.patch.object(
            reranker_client, "rerank", return_value=exp_rerank_response
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
