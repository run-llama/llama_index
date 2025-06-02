import json
import aioboto3.session
import pytest

import aioboto3
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
