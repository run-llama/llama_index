import os
from llama_index.core.base.llms.types import ImageBlock, TextBlock
import pytest
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    CompletionResponse,
    ImageBlock,
    TextBlock,
)
from llama_index.core.callbacks import CallbackManager
from PIL import Image
import io
import numpy as np

# Expected values
EXP_RESPONSE = "Test"
EXP_STREAM_RESPONSE = ["Test ", "value"]
EXP_MAX_TOKENS = 100
EXP_TEMPERATURE = 0.7
EXP_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"
EXP_GUARDRAIL_ID = "IDENTIFIER"
EXP_GUARDRAIL_VERSION = "DRAFT"
EXP_GUARDRAIL_TRACE = "ENABLED"

# Reused chat message and prompt
messages = [ChatMessage(role=MessageRole.USER, content="Test")]
prompt = "Test"

# --- Integration Tests ---
# These tests will call the actual AWS Bedrock API.
# They will be skipped if AWS credentials are not found in the environment.
needs_aws_creds = pytest.mark.skipif(
    os.getenv("AWS_ACCESS_KEY_ID") is None
    or os.getenv("AWS_SECRET_ACCESS_KEY") is None
    or os.getenv("AWS_REGION") is None,
    reason="AWS credentials not found in environment, skipping integration test",
)


@pytest.fixture(scope="module")
def temp_image_bytes():
    """Generate a 100x100 red image directly in memory and return its bytes."""
    width, height = 100, 100
    red_array = np.zeros((height, width, 3), dtype=np.uint8)
    red_array[:, :, 0] = 255

    img = Image.fromarray(red_array)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer.read()


@pytest.fixture(scope="module")
def bedrock_converse_integration():
    """Create a BedrockConverse instance for integration tests with proper credentials."""
    return BedrockConverse(
        model=EXP_MODEL,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        max_tokens=EXP_MAX_TOKENS,
    )


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


@needs_aws_creds
def test_bedrock_converse_integration_chat_text_only(bedrock_converse_integration):
    """Test a simple text chat integration with Bedrock Converse."""
    llm = bedrock_converse_integration
    messages = [
        ChatMessage(role=MessageRole.USER, content="Write a short sonnet about clouds.")
    ]
    response = llm.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert isinstance(response.message.content, str)
    assert len(response.message.content) > 5


@needs_aws_creds
def test_bedrock_converse_integration_chat_multimodal(
    temp_image_bytes, bedrock_converse_integration
):
    """Test multimodal chat (text + image) integration with Bedrock Converse."""
    llm = bedrock_converse_integration
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            blocks=[ImageBlock(image=temp_image_bytes, image_mimetype="image/png")],
        ),
        ChatMessage(
            role=MessageRole.USER,
            blocks=[TextBlock(text="What color do you see in the image above?")],
        ),
    ]

    response = llm.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert isinstance(response.message.content, str)
    assert "red" in response.message.content.lower()


@needs_aws_creds
@pytest.mark.asyncio()
async def test_bedrock_converse_integration_achat_text_only(
    bedrock_converse_integration,
):
    """Test async text chat integration."""
    llm = bedrock_converse_integration
    messages = [
        ChatMessage(role=MessageRole.USER, content="What is the capital of France?")
    ]
    response = await llm.achat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert isinstance(response.message.content, str)
    assert "paris" in response.message.content.lower()


@needs_aws_creds
@pytest.mark.asyncio()
async def test_bedrock_converse_integration_achat_multimodal(
    temp_image_bytes, bedrock_converse_integration
):
    """Test async multimodal chat integration."""
    llm = bedrock_converse_integration

    # Use the red image data from temp_image_bytes fixture
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            blocks=[ImageBlock(image=temp_image_bytes, image_mimetype="image/png")],
        ),
        ChatMessage(
            role=MessageRole.USER,
            blocks=[TextBlock(text="Describe the image provided above briefly.")],
        ),
    ]

    response = await llm.achat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert isinstance(response.message.content, str)
    assert len(response.message.content) > 5


@needs_aws_creds
def test_bedrock_converse_integration_stream_chat(bedrock_converse_integration):
    """Test streaming chat integration with Bedrock Converse."""
    llm = bedrock_converse_integration
    messages = [ChatMessage(role=MessageRole.USER, content="Count from 1 to 5 slowly.")]

    response_stream = llm.stream_chat(messages)
    chunks = []
    for response in response_stream:
        chunks.append(response.delta)

    assert len(chunks) > 1
    combined = "".join(chunks)
    assert len(combined) > 5


@needs_aws_creds
def test_bedrock_converse_integration_stream_chat_multimodal(
    temp_image_bytes, bedrock_converse_integration
):
    """Test streaming multimodal chat integration with Bedrock Converse."""
    llm = bedrock_converse_integration
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            blocks=[ImageBlock(image=temp_image_bytes, image_mimetype="image/png")],
        ),
        ChatMessage(
            role=MessageRole.USER,
            blocks=[TextBlock(text="Describe this image in a few words.")],
        ),
    ]

    response_stream = llm.stream_chat(messages)
    chunks = []
    for response in response_stream:
        chunks.append(response.delta)

    assert len(chunks) > 1
    combined = "".join(chunks)
    assert len(combined) > 5


@needs_aws_creds
@pytest.mark.asyncio()
async def test_bedrock_converse_integration_astream_chat(bedrock_converse_integration):
    """Test async streaming chat integration with Bedrock Converse."""
    llm = bedrock_converse_integration

    messages = [
        ChatMessage(role=MessageRole.USER, content="Name three famous scientists.")
    ]

    response_stream = await llm.astream_chat(messages)
    chunks = []
    async for response in response_stream:
        chunks.append(response.delta)

    assert len(chunks) > 1
    combined = "".join(chunks)
    assert len(combined) > 5


@needs_aws_creds
@pytest.mark.asyncio()
async def test_bedrock_converse_integration_astream_chat_multimodal(
    temp_image_bytes, bedrock_converse_integration
):
    """Test async streaming multimodal chat integration with Bedrock Converse."""
    llm = bedrock_converse_integration
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            blocks=[ImageBlock(image=temp_image_bytes, image_mimetype="image/png")],
        ),
        ChatMessage(
            role=MessageRole.USER,
            blocks=[TextBlock(text="What do you see in this image?")],
        ),
    ]

    response_stream = await llm.astream_chat(messages)
    chunks = []
    async for response in response_stream:
        chunks.append(response.delta)

    assert len(chunks) > 1
    combined = "".join(chunks)
    assert len(combined) > 5
