import json
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from botocore.response import StreamingBody
from llama_index.embeddings.bedrock import BedrockEmbedding, Models

try:
    import aioboto3

    HAS_AIOBOTO3 = True
except ImportError:
    HAS_AIOBOTO3 = False

EXP_REQUEST = "foo bar baz"
EXP_RESPONSE = {
    "embedding": [0.017410278, 0.040924072, -0.007507324, 0.09429932, 0.015304565]
}


# --- Sync mock (used when aioboto3 is NOT installed) ---


def _make_sync_mock_client(response_body=None):
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


# --- Async mock (used when aioboto3 IS installed) ---


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


# --- Fixtures ---


@pytest.fixture()
def bedrock_embedding_sync():
    """Embedding instance using sync client (to_thread fallback path)."""
    return BedrockEmbedding(
        model_name=Models.TITAN_EMBEDDING,
        client=_make_sync_mock_client(),
    )


@pytest.fixture()
def mock_aioboto3_session(monkeypatch):
    if HAS_AIOBOTO3:
        monkeypatch.setattr("aioboto3.Session", AsyncMockSession)


@pytest.fixture()
def bedrock_embedding_async(mock_aioboto3_session):
    """Embedding instance using async session (native aioboto3 path)."""
    return BedrockEmbedding(
        model_name=Models.TITAN_EMBEDDING,
        client=_make_sync_mock_client(),  # sync client still needed for init
    )


# --- Tests: to_thread fallback (no aioboto3) ---


@pytest.mark.asyncio
@pytest.mark.skipif(HAS_AIOBOTO3, reason="testing to_thread fallback only")
async def test_aget_text_embedding_fallback(bedrock_embedding_sync):
    response = await bedrock_embedding_sync._aget_text_embedding(EXP_REQUEST)
    assert response == EXP_RESPONSE["embedding"]


@pytest.mark.asyncio
@pytest.mark.skipif(HAS_AIOBOTO3, reason="testing to_thread fallback only")
async def test_aget_query_embedding_fallback(bedrock_embedding_sync):
    response = await bedrock_embedding_sync._aget_query_embedding(EXP_REQUEST)
    assert response == EXP_RESPONSE["embedding"]


# --- Tests: native async (with aioboto3) ---


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AIOBOTO3, reason="requires aioboto3")
async def test_aget_text_embedding_native(bedrock_embedding_async):
    response = await bedrock_embedding_async._aget_text_embedding(EXP_REQUEST)
    assert response == EXP_RESPONSE["embedding"]


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AIOBOTO3, reason="requires aioboto3")
async def test_aget_query_embedding_native(bedrock_embedding_async):
    response = await bedrock_embedding_async._aget_query_embedding(EXP_REQUEST)
    assert response == EXP_RESPONSE["embedding"]


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AIOBOTO3, reason="requires aioboto3")
async def test_application_inference_profile_in_invoke_model_request(
    mock_aioboto3_session,
):
    from unittest import mock

    mock_client = _make_sync_mock_client()
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

    with mock.patch.object(
        AsyncMockClient, "invoke_model", wraps=AsyncMockClient().invoke_model
    ) as patched_invoke:
        await bedrock_embedding._aget_text_embedding(EXP_REQUEST)
        assert patched_invoke.called
        assert (
            patched_invoke.call_args.kwargs["modelId"]
            == application_inference_profile_arn
        )
