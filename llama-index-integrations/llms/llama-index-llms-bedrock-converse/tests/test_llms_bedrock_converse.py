import pytest
import asyncio
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    CompletionResponse,
)
from llama_index.core.callbacks import CallbackManager


class MockThrottlingException(Exception):
    pass


class MockExceptions:
    ThrottlingException = MockThrottlingException


class AsyncMockClient:
    def __init__(self) -> "AsyncMockClient":
        self.exceptions = MockExceptions()

    async def __aenter__(self) -> "AsyncMockClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    async def converse(self, *args, **kwargs):
        return {"output": {"message": {"content": [{"text": "Async mock response"}]}}}

    async def converse_stream(self, *args, **kwargs):
        async def stream_generator():
            yield {"contentBlockDelta": {"delta": {"text": "Async mock "}}}
            await asyncio.sleep(0.1)
            yield {"contentBlockDelta": {"delta": {"text": "streamed "}}}
            await asyncio.sleep(0.1)
            yield {"contentBlockDelta": {"delta": {"text": "response"}}}

        return {"stream": stream_generator()}


class MockClient:
    def __init__(self) -> "MockClient":
        self.exceptions = MockExceptions()

    def converse(self, *args, **kwargs):
        return {"output": {"message": {"content": [{"text": "Mock response"}]}}}

    def converse_stream(self, *args, **kwargs):
        return {"stream": self._stream_generator()}

    def _stream_generator(self):
        yield {"contentBlockDelta": {"delta": {"text": "Mock "}}}
        yield {"contentBlockDelta": {"delta": {"text": "streamed "}}}
        yield {"contentBlockDelta": {"delta": {"text": "response"}}}


class MockAsyncSession:
    def __init__(self, *args, **kwargs) -> "MockAsyncSession":
        pass

    def client(self, *args, **kwargs):
        return AsyncClientContextManager()


class AsyncClientContextManager:
    async def __aenter__(self) -> AsyncMockClient:
        return AsyncMockClient()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


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
        model="anthropic.claude-v2",
        max_tokens=100,
        temperature=0.7,
        callback_manager=CallbackManager(),
        system_prompt=None,
    )


def test_init(bedrock_converse):
    assert bedrock_converse.model == "anthropic.claude-v2"
    assert bedrock_converse.max_tokens == 100
    assert bedrock_converse.temperature == 0.7
    assert bedrock_converse._client is not None


def test_chat(bedrock_converse):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?"),
    ]

    response = bedrock_converse.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content == "Mock response"
    assert isinstance(response.raw, dict)
    assert "output" in response.raw


def test_complete(bedrock_converse):
    prompt = "Once upon a time,"

    response = bedrock_converse.complete(prompt)

    assert isinstance(response, CompletionResponse)
    assert response.text == "Mock response"
    assert isinstance(response.raw, dict)
    assert "output" in response.raw
    assert set(response.additional_kwargs.keys()) == {
        "status",
        "tool_call_id",
        "tool_calls",
    }
    assert response.additional_kwargs["status"] == []
    assert response.additional_kwargs["tool_call_id"] == []
    assert response.additional_kwargs["tool_calls"] == []


def test_stream_chat(bedrock_converse):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?"),
    ]

    response_stream = bedrock_converse.stream_chat(messages)

    responses = list(response_stream)

    assert len(responses) == 3
    assert all(isinstance(r, ChatResponse) for r in responses)
    assert responses[0].delta == "Mock "
    assert responses[1].delta == "streamed "
    assert responses[2].delta == "response"

    final_response = responses[-1].message
    assert final_response.role == MessageRole.ASSISTANT
    assert final_response.content == "Mock streamed response"
    assert isinstance(final_response.additional_kwargs, dict)
    assert set(final_response.additional_kwargs.keys()) == {
        "tool_calls",
        "tool_call_id",
        "status",
    }
    assert all(
        final_response.additional_kwargs[key] == []
        for key in final_response.additional_kwargs
    )


@pytest.mark.asyncio()
async def test_achat(bedrock_converse):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?"),
    ]

    response = await bedrock_converse.achat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content == "Async mock response"
    assert isinstance(response.raw, dict)
    assert "output" in response.raw
    assert isinstance(response.message.additional_kwargs, dict)
    assert set(response.message.additional_kwargs.keys()) == {
        "tool_calls",
        "tool_call_id",
        "status",
    }
    assert all(
        response.message.additional_kwargs[key] == []
        for key in response.message.additional_kwargs
    )


@pytest.mark.asyncio()
async def test_astream_chat(bedrock_converse):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?"),
    ]

    response_stream = await bedrock_converse.astream_chat(messages)

    responses = []
    async for response in response_stream:
        responses.append(response)

    assert len(responses) == 3
    assert all(isinstance(r, ChatResponse) for r in responses)
    assert responses[0].delta == "Async mock "
    assert responses[1].delta == "streamed "
    assert responses[2].delta == "response"

    final_response = responses[-1].message
    assert final_response.role == MessageRole.ASSISTANT
    assert final_response.content == "Async mock streamed response"
    assert isinstance(final_response.additional_kwargs, dict)
    assert set(final_response.additional_kwargs.keys()) == {
        "tool_calls",
        "tool_call_id",
        "status",
    }
    assert all(
        final_response.additional_kwargs[key] == []
        for key in final_response.additional_kwargs
    )


@pytest.mark.asyncio()
async def test_acomplete(bedrock_converse):
    prompt = "Once upon a time,"

    response = await bedrock_converse.acomplete(prompt)

    assert isinstance(response, CompletionResponse)
    assert response.text == "Async mock response"
    assert isinstance(response.raw, dict)
    assert "output" in response.raw
    assert set(response.additional_kwargs.keys()) == {
        "status",
        "tool_call_id",
        "tool_calls",
    }
    assert response.additional_kwargs["status"] == []
    assert response.additional_kwargs["tool_call_id"] == []
    assert response.additional_kwargs["tool_calls"] == []


@pytest.mark.asyncio()
async def test_astream_complete(bedrock_converse):
    prompt = "Once upon a time,"

    response_stream = await bedrock_converse.astream_complete(prompt)

    responses = []
    async for response in response_stream:
        responses.append(response)

    assert len(responses) == 3
    assert all(isinstance(r, CompletionResponse) for r in responses)
    assert responses[0].delta == "Async mock "
    assert responses[1].delta == "streamed "
    assert responses[2].delta == "response"

    final_response = responses[-1]
    assert final_response.text == "Async mock streamed response"
    assert isinstance(final_response.raw, dict)
    assert "stream" in final_response.raw
    assert isinstance(final_response.raw["stream"], object)
    assert set(final_response.additional_kwargs.keys()) == {
        "status",
        "tool_call_id",
        "tool_calls",
    }
    assert all(
        final_response.additional_kwargs[key] == []
        for key in final_response.additional_kwargs
    )
