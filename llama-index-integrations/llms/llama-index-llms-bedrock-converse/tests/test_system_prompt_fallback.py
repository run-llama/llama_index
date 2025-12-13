import pytest
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.base.llms.types import ChatMessage, MessageRole


class MockClient:
    """Mock Bedrock client for testing."""

    def __init__(self):
        self.exceptions = type(
            "Exceptions",
            (),
            {"ThrottlingException": type("ThrottlingException", (Exception,), {})},
        )
        self.last_system_param = None

    def converse(self, **kwargs):
        """Mock converse method that tracks the system parameter."""
        self.last_system_param = kwargs.get("system")

        return {
            "ResponseMetadata": {"RequestId": "test-request-id"},
            "output": {
                "message": {"role": "assistant", "content": [{"text": "Test response"}]}
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 10,
                "totalTokens": 110,
            },
            "metrics": {"latencyMs": 100},
        }


@pytest.fixture()
def mock_boto3_session(monkeypatch):
    """Mock boto3 session to return our MockClient."""

    def mock_client(*args, **kwargs):
        return MockClient()

    monkeypatch.setattr("boto3.Session.client", mock_client)


@pytest.fixture()
def bedrock_converse_with_system_prompt(mock_boto3_session):
    """Create BedrockConverse with system_prompt and system_prompt_caching."""
    return BedrockConverse(
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        region_name="us-east-1",
        system_prompt="You are a helpful AI assistant.",
        system_prompt_caching=True,
    )


def test_system_prompt_fallback_in_chat(bedrock_converse_with_system_prompt):
    """Test that system_prompt is used when messages don't contain SYSTEM role."""
    llm = bedrock_converse_with_system_prompt
    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

    response = llm.chat(messages)

    # Verify system prompt was sent to API
    assert llm._client.last_system_param is not None
    assert len(llm._client.last_system_param) > 0

    # System should be list format with text and cachePoint
    assert isinstance(llm._client.last_system_param, list)
    assert any(
        "helpful AI assistant" in str(item) for item in llm._client.last_system_param
    )
    # Should have cachePoint added due to system_prompt_caching=True
    assert any(
        "cachePoint" in item
        for item in llm._client.last_system_param
        if isinstance(item, dict)
    )

    assert response.message.content == "Test response"


def test_message_system_prompt_overrides_param(bedrock_converse_with_system_prompt):
    """Test that SYSTEM message in messages takes precedence."""
    llm = bedrock_converse_with_system_prompt
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="Override system prompt"),
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]

    response = llm.chat(messages)

    # System should be from messages, not self.system_prompt
    system_param = llm._client.last_system_param
    assert system_param is not None

    # Should contain the override text
    system_str = str(system_param)
    assert "Override system prompt" in system_str


def test_empty_system_prompt_no_fallback(mock_boto3_session):
    """Test that when both are empty, no system is sent."""
    llm = BedrockConverse(
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        region_name="us-east-1",
    )

    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
    llm.chat(messages)

    # When both are empty, system should be None or empty
    system_param = llm._client.last_system_param
    assert system_param is None or len(system_param) == 0


def test_system_prompt_with_caching_creates_cache_point(
    bedrock_converse_with_system_prompt,
):
    """Test that cachePoint is added when system_prompt_caching=True."""
    llm = bedrock_converse_with_system_prompt
    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

    llm.chat(messages)

    # System should be a list with cachePoint appended
    system_param = llm._client.last_system_param
    assert isinstance(system_param, list)

    # Last item should be a cachePoint
    assert any("cachePoint" in item for item in system_param if isinstance(item, dict))
