import pytest
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    CompletionResponse,
)
from llama_index.core.callbacks import CallbackManager

# Expected values
EXP_RESPONSE = "Test"
EXP_STREAM_RESPONSE = ["Test ", "value"]
EXP_MAX_TOKENS = 100
EXP_TEMPERATURE = 0.7
EXP_MODEL = "anthropic.claude-v2"
EXP_GUARDRAIL_ID = "IDENTIFIER"
EXP_GUARDRAIL_VERSION = "DRAFT"
EXP_GUARDRAIL_TRACE = "ENABLED"

# Reused chat message and prompt
messages = [ChatMessage(role=MessageRole.USER, content="Test")]
prompt = "Test"


class MockExceptions:
    class ThrottlingException(Exception):
        pass


class AsyncMockClient:
    def __init__(self) -> "AsyncMockClient":
        self.exceptions = MockExceptions()

    async def __aenter__(self) -> "AsyncMockClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    async def converse(self, *args, **kwargs):
        return {"output": {"message": {"content": [{"text": EXP_RESPONSE}]}}}

    async def converse_stream(self, *args, **kwargs):
        async def stream_generator():
            for element in EXP_STREAM_RESPONSE:
                yield {"contentBlockDelta": {"delta": {"text": element}}}

        return {"stream": stream_generator()}


class MockClient:
    def __init__(self) -> "MockClient":
        self.exceptions = MockExceptions()

    def converse(self, *args, **kwargs):
        return {"output": {"message": {"content": [{"text": EXP_RESPONSE}]}}}

    def converse_stream(self, *args, **kwargs):
        def stream_generator():
            for element in EXP_STREAM_RESPONSE:
                yield {"contentBlockDelta": {"delta": {"text": element}}}

        return {"stream": stream_generator()}


class MockAsyncSession:
    def __init__(self, *args, **kwargs) -> "MockAsyncSession":
        pass

    def client(self, *args, **kwargs):
        return AsyncMockClient()


@pytest.fixture()
def mock_boto3_session(monkeypatch):
    def mock_client(*args, **kwargs):
        return MockClient()

    monkeypatch.setattr("boto3.Session.client", mock_client)


@pytest.fixture()
def mock_aioboto3_session(monkeypatch):
    monkeypatch.setattr("aioboto3.Session", MockAsyncSession)


@pytest.fixture()
def bedrock_converse(mock_boto3_session, mock_aioboto3_session):
    return BedrockConverse(
        model=EXP_MODEL,
        max_tokens=EXP_MAX_TOKENS,
        temperature=EXP_TEMPERATURE,
        guardrail_identifier=EXP_GUARDRAIL_ID,
        guardrail_version=EXP_GUARDRAIL_VERSION,
        trace=EXP_GUARDRAIL_TRACE,
        callback_manager=CallbackManager(),
    )


def test_init(bedrock_converse):
    assert bedrock_converse.model == EXP_MODEL
    assert bedrock_converse.max_tokens == EXP_MAX_TOKENS
    assert bedrock_converse.temperature == EXP_TEMPERATURE
    assert bedrock_converse._client is not None


def test_chat(bedrock_converse):
    response = bedrock_converse.chat(messages)

    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content == EXP_RESPONSE


def test_complete(bedrock_converse):
    response = bedrock_converse.complete(prompt)

    assert isinstance(response, CompletionResponse)
    assert response.text == EXP_RESPONSE
    assert response.additional_kwargs["status"] == []
    assert response.additional_kwargs["tool_call_id"] == []
    assert response.additional_kwargs["tool_calls"] == []


def test_stream_chat(bedrock_converse):
    response_stream = bedrock_converse.stream_chat(messages)

    for response in response_stream:
        assert response.message.role == MessageRole.ASSISTANT
        assert response.delta in EXP_STREAM_RESPONSE


@pytest.mark.asyncio()
async def test_achat(bedrock_converse):
    response = await bedrock_converse.achat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content == EXP_RESPONSE


@pytest.mark.asyncio()
async def test_astream_chat(bedrock_converse):
    response_stream = await bedrock_converse.astream_chat(messages)

    responses = []
    async for response in response_stream:
        assert response.message.role == MessageRole.ASSISTANT
        assert response.delta in EXP_STREAM_RESPONSE


@pytest.mark.asyncio()
async def test_acomplete(bedrock_converse):
    response = await bedrock_converse.acomplete(prompt)

    assert isinstance(response, CompletionResponse)
    assert response.text == EXP_RESPONSE
    assert response.additional_kwargs["status"] == []
    assert response.additional_kwargs["tool_call_id"] == []
    assert response.additional_kwargs["tool_calls"] == []


@pytest.mark.asyncio()
async def test_astream_complete(bedrock_converse):
    response_stream = await bedrock_converse.astream_complete(prompt)

    async for response in response_stream:
        assert response.delta in EXP_STREAM_RESPONSE
