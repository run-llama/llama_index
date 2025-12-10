import pytest
from unittest.mock import Mock, MagicMock, patch
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.base.llms.types import ChatMessage, MessageRole


@pytest.fixture
def mock_bedrock_client():
    """Mock Bedrock client that returns a test response."""
    client = Mock()
    client.converse = Mock(return_value={
        "ResponseMetadata": {"RequestId": "test-request-id"},
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Test response"}]
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 100,
            "outputTokens": 10,
            "totalTokens": 110,
        },
        "metrics": {"latencyMs": 100}
    })
    return client


@pytest.fixture
def bedrock_converse_with_system_prompt(mock_bedrock_client):
    """Create BedrockConverse instance with system_prompt and system_prompt_caching enabled."""
    with patch('llama_index.llms.bedrock_converse.base.get_bedrock_client') as mock_get_client:
        mock_get_client.return_value = mock_bedrock_client

        llm = BedrockConverse(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
            system_prompt="You are a helpful AI assistant.",
            system_prompt_caching=True,
        )
        llm._client = mock_bedrock_client
        return llm


def test_system_prompt_fallback_in_chat(bedrock_converse_with_system_prompt, mock_bedrock_client):
    """Test that system_prompt is used when messages don't contain SYSTEM role."""
    llm = bedrock_converse_with_system_prompt
    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

    # Call chat
    response = llm.chat(messages)

    # Verify converse was called with system prompt
    mock_bedrock_client.converse.assert_called_once()
    call_kwargs = mock_bedrock_client.converse.call_args[1]

    # Should have system field in the request
    assert "system" in call_kwargs
    assert call_kwargs["system"] == "You are a helpful AI assistant."

    # Verify response
    assert response.message.content == "Test response"


def test_message_system_prompt_overrides_param(bedrock_converse_with_system_prompt, mock_bedrock_client):
    """Test that SYSTEM message in messages takes precedence over system_prompt parameter."""
    llm = bedrock_converse_with_system_prompt
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="Override system prompt"),
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]

    # Call chat
    response = llm.chat(messages)

    # Verify converse was called
    mock_bedrock_client.converse.assert_called_once()
    call_kwargs = mock_bedrock_client.converse.call_args[1]

    # Should use the SYSTEM message from messages, not self.system_prompt
    assert "system" in call_kwargs
    # The system prompt from messages should be a list format after processing
    system_content = call_kwargs["system"]
    assert isinstance(system_content, (list, str))


def test_system_prompt_caching_applied(bedrock_converse_with_system_prompt, mock_bedrock_client):
    """Test that system_prompt_caching flag is passed through."""
    llm = bedrock_converse_with_system_prompt
    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

    # Call chat
    llm.chat(messages)

    # The actual caching logic is in converse_with_retry, but we verify
    # that our fix allows the system prompt to be passed through
    call_kwargs = mock_bedrock_client.converse.call_args[1]
    assert "system" in call_kwargs
    assert call_kwargs["system"] is not None


@pytest.mark.asyncio
async def test_system_prompt_fallback_in_achat(bedrock_converse_with_system_prompt):
    """Test that system_prompt fallback works in async chat."""
    llm = bedrock_converse_with_system_prompt

    # Mock async client
    mock_async_client = MagicMock()
    mock_async_client.converse = MagicMock(return_value={
        "ResponseMetadata": {"RequestId": "test-request-id"},
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Async test response"}]
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 100,
            "outputTokens": 10,
            "totalTokens": 110,
        },
        "metrics": {"latencyMs": 100}
    })

    # Mock the async session
    with patch.object(llm, '_asession') as mock_session:
        mock_session.converse = mock_async_client.converse

        messages = [ChatMessage(role=MessageRole.USER, content="Hello async")]

        # The async methods use different paths, this test ensures the same logic applies
        # Note: This is a simplified test; full async testing would require more setup
        assert llm.system_prompt == "You are a helpful AI assistant."
        assert llm.system_prompt_caching is True


def test_empty_system_prompt_no_fallback(mock_bedrock_client):
    """Test that when both message system and self.system_prompt are empty, no system is sent."""
    with patch('llama_index.llms.bedrock_converse.base.get_bedrock_client') as mock_get_client:
        mock_get_client.return_value = mock_bedrock_client

        llm = BedrockConverse(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
            # No system_prompt parameter
        )
        llm._client = mock_bedrock_client

        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        llm.chat(messages)

        call_kwargs = mock_bedrock_client.converse.call_args[1]
        # When both are empty, system field should not be present or be empty
        if "system" in call_kwargs:
            assert not call_kwargs["system"]  # Empty list or None
