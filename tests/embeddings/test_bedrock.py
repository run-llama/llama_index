import json
from io import BytesIO
from unittest import TestCase

import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber
from llama_index.embeddings.bedrock import BedrockEmbedding, Models


class TestBedrockEmbedding(TestCase):
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    bedrock_stubber = Stubber(bedrock_client)

    def test_get_text_embedding_titan(self) -> None:
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
            model=Models.TITAN_EMBEDDING,
            client=self.bedrock_client,
        )

        self.bedrock_stubber.activate()
        embedding = bedrock_embedding.get_text_embedding(text="foo bar baz")
        self.bedrock_stubber.deactivate()

        self.bedrock_stubber.assert_no_pending_responses()
        self.assertEqual(embedding, mock_response["embedding"])

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
