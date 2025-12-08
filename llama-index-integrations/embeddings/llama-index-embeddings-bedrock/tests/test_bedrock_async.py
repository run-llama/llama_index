import json
from unittest import mock

import aioboto3
import aioboto3.session
import pytest
from llama_index.embeddings.bedrock import BedrockEmbedding, Models

EXP_REQUEST = "foo bar baz"
EXP_RESPONSE = {
    "embedding": [
        0.017410278,
        0.040924072,
        -0.007507324,
        0.09429932,
        0.015304565,
    ]
}


class AsyncMockStreamReader:
    async def read(self):
        return json.dumps(EXP_RESPONSE).encode()


class AsyncMockClient:
    async def __aenter__(self) -> "AsyncMockClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    async def invoke_model(self, *args, **kwargs):
        return {"contentType": "application/json", "body": AsyncMockStreamReader()}


class AsyncMockSession:
    def __init__(self, *args, **kwargs) -> "AsyncMockSession":
        pass

    def client(self, *args, **kwargs):
        return AsyncMockClient()


@pytest.fixture()
def mock_aioboto3_session(monkeypatch):
    monkeypatch.setattr("aioboto3.Session", AsyncMockSession)


@pytest.fixture()
def bedrock_embedding(mock_aioboto3_session):
    return BedrockEmbedding(
        model_name=Models.TITAN_EMBEDDING,
        client=aioboto3.Session().client("bedrock-runtime", region_name="us-east-1"),
    )


@pytest.mark.asyncio
async def test_aget_text_embedding(bedrock_embedding):
    response = await bedrock_embedding._aget_text_embedding(EXP_REQUEST)
    assert response == EXP_RESPONSE["embedding"]


@pytest.mark.asyncio
async def test_application_inference_profile_in_invoke_model_request(
    mock_aioboto3_session,
):
    client = aioboto3.Session().client("bedrock-runtime", region_name="us-east-1")
    model_name = Models.TITAN_EMBEDDING_V2_0
    application_inference_profile_arn = "arn:aws:bedrock:us-east-1:012345678901:application-inference-profile/testProfileId"

    bedrock_embedding = BedrockEmbedding(
        model_name=model_name,
        application_inference_profile_arn=application_inference_profile_arn,
        client=client,
    )
    assert bedrock_embedding.model_name == model_name
    assert (
        bedrock_embedding.application_inference_profile_arn
        == application_inference_profile_arn
    )

    with mock.patch.object(
        AsyncMockClient, "invoke_model", wraps=client.invoke_model
    ) as patched_invoke:
        await bedrock_embedding._aget_text_embedding(EXP_REQUEST)

        assert patched_invoke.called
        assert (
            patched_invoke.call_args.kwargs["modelId"]
            == application_inference_profile_arn
        )
