import json
from io import BytesIO
from unittest import TestCase, mock

import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber
from llama_index.embeddings.bedrock import BedrockEmbedding, Models


class TestBedrockEmbedding(TestCase):
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    bedrock_stubber = Stubber(bedrock_client)

    def _test_get_text_embedding(self, model, titan_body_kwargs=None):
        mock_response = {
            "embedding": [
                0.017410278,
                0.040924072,
                -0.007507324,
                0.09429932,
                0.015304565,
            ]
        }

        mock_stream = BytesIO(json.dumps(mock_response).encode())

        self.bedrock_stubber.add_response(
            "invoke_model",
            {
                "contentType": "application/json",
                "body": StreamingBody(mock_stream, len(json.dumps(mock_response))),
            },
        )

        bedrock_embedding = BedrockEmbedding(
            model=model,
            client=self.bedrock_client,
            titan_body_kwargs=titan_body_kwargs,
        )

        self.bedrock_stubber.activate()
        embedding = bedrock_embedding.get_text_embedding(text="foo bar baz")
        self.bedrock_stubber.deactivate()

        self.bedrock_stubber.assert_no_pending_responses()
        self.assertEqual(embedding, mock_response["embedding"])

    def test_get_text_embedding_titan_v1(self) -> None:
        self._test_get_text_embedding(Models.TITAN_EMBEDDING)

    def test_get_text_embedding_titan_v2(self) -> None:
        self._test_get_text_embedding(Models.TITAN_EMBEDDING, titan_body_kwargs={
            "dimensions": 512,
            "normalize": True
        })

    def test_get_text_embedding_cohere(self) -> None:
        mock_response = {
            "embeddings": [
                [0.017410278, 0.040924072, -0.007507324, 0.09429932, 0.015304565]
            ]
        }

        mock_stream = BytesIO(json.dumps(mock_response).encode())

        self.bedrock_stubber.add_response(
            "invoke_model",
            {
                "contentType": "application/json",
                "body": StreamingBody(mock_stream, len(json.dumps(mock_response))),
            },
        )

        bedrock_embedding = BedrockEmbedding(
            model=Models.COHERE_EMBED_ENGLISH_V3,
            client=self.bedrock_client,
        )

        self.bedrock_stubber.activate()
        embedding = bedrock_embedding.get_text_embedding(text="foo bar baz")
        self.bedrock_stubber.deactivate()

        self.bedrock_stubber.assert_no_pending_responses()
        self.assertEqual(embedding, mock_response["embeddings"][0])

    def test_get_text_embedding_batch_cohere(self) -> None:
        mock_response = {
            "embeddings": [
                [0.017410278, 0.040924072, -0.007507324, 0.09429932, 0.015304565],
                [0.017410278, 0.040924072, -0.007507324, 0.09429932, 0.015304565],
            ]
        }
        mock_request = ["foo bar baz", "foo baz bar"]

        mock_stream = BytesIO(json.dumps(mock_response).encode())

        self.bedrock_stubber.add_response(
            "invoke_model",
            {
                "contentType": "application/json",
                "body": StreamingBody(mock_stream, len(json.dumps(mock_response))),
            },
        )

        bedrock_embedding = BedrockEmbedding(
            model=Models.COHERE_EMBED_ENGLISH_V3,
            client=self.bedrock_client,
        )

        self.bedrock_stubber.activate()
        embedding = bedrock_embedding.get_text_embedding_batch(texts=mock_request)

        self.bedrock_stubber.deactivate()

        self.assertEqual(len(embedding), 2)

        for i in range(2):
            self.assertEqual(embedding[i], mock_response["embeddings"][i])
