import json
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from botocore.response import StreamingBody
from llama_index.embeddings.bedrock import BedrockEmbedding, Models

EXP_REQUEST = "foo bar baz"
EXP_RESPONSE = {
    "embedding": [0.017410278, 0.040924072, -0.007507324, 0.09429932, 0.015304565]
}


def _make_mock_client(response_body=None):
    """Create a mock boto3 client that returns the given response body."""
    if response_body is None:
        response_body = EXP_RESPONSE
    encoded = json.dumps(response_body).encode()
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {
        "contentType": "application/json",
        "body": StreamingBody(BytesIO(encoded), len(encoded)),
    }
    return mock_client


@pytest.fixture()
def bedrock_embedding():
    return BedrockEmbedding(
        model_name=Models.TITAN_EMBEDDING,
        client=_make_mock_client(),
    )


@pytest.mark.asyncio
async def test_aget_text_embedding(bedrock_embedding):
    response = await bedrock_embedding._aget_text_embedding(EXP_REQUEST)
    assert response == EXP_RESPONSE["embedding"]


@pytest.mark.asyncio
async def test_aget_query_embedding(bedrock_embedding):
    response = await bedrock_embedding._aget_query_embedding(EXP_REQUEST)
    assert response == EXP_RESPONSE["embedding"]


@pytest.mark.asyncio
async def test_application_inference_profile_in_invoke_model_request():
    mock_client = _make_mock_client()
    model_name = Models.TITAN_EMBEDDING_V2_0
    application_inference_profile_arn = "arn:aws:bedrock:us-east-1:012345678901:application-inference-profile/testProfileId"

    bedrock_embedding = BedrockEmbedding(
        model_name=model_name,
        application_inference_profile_arn=application_inference_profile_arn,
        client=mock_client,
    )
    assert bedrock_embedding.model_name == model_name
    assert (
        bedrock_embedding.application_inference_profile_arn
        == application_inference_profile_arn
    )

    await bedrock_embedding._aget_text_embedding(EXP_REQUEST)

    mock_client.invoke_model.assert_called_once()
    assert (
        mock_client.invoke_model.call_args.kwargs["modelId"]
        == application_inference_profile_arn
    )
