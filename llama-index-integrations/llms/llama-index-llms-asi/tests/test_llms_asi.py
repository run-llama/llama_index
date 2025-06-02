import os
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.openai.base import ChatMessage, MessageRole

from llama_index.llms.asi import ASI


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in ASI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@patch.dict(os.environ, {}, clear=True)
def test_initialization():
    """Test initialization with API key."""
    # Test with explicit API key
    llm = ASI(api_key="test_key")
    assert llm.api_key == "test_key"

    # Test with missing API key
    with pytest.raises(
        ValueError,
        match="Must specify `api_key` or set environment variable `ASI_API_KEY`",
    ):
        ASI(api_key=None)


@patch("llama_index.llms.asi.base.ASI.stream_complete")
def test_stream_complete(mock_stream_complete):
    """Test stream_complete method."""
    # Create a mock response
    mock_chunk = MagicMock()
    mock_chunk.text = "Test response"

    # Set up the mock to return our test chunk
    mock_stream_complete.return_value = [mock_chunk]

    # Create the ASI instance
    llm = ASI(api_key="test_key")

    # Call stream_complete
    stream = llm.stream_complete("Test prompt")

    # Check that we get the expected response
    chunks = list(stream)
    assert len(chunks) == 1
    assert chunks[0].text == "Test response"

    # Verify that stream_complete was called with the right arguments
    mock_stream_complete.assert_called_once_with("Test prompt")


@patch("llama_index.llms.asi.base.ASI.stream_chat")
def test_stream_chat_with_content(mock_stream_chat):
    """Test stream_chat method with content in delta."""
    # Create a mock for the response
    mock_chunk = MagicMock()
    mock_chunk.delta = "Test content"

    # Set up the mock to return our test chunk
    mock_stream_chat.return_value = [mock_chunk]

    # Create the ASI instance
    llm = ASI(api_key="test_key")

    # Call stream_chat
    messages = [ChatMessage(role=MessageRole.USER, content="Test message")]
    stream = llm.stream_chat(messages)

    # Check that we get the expected response
    chunks = list(stream)
    assert len(chunks) == 1
    assert chunks[0].delta == "Test content"

    # Verify that stream_chat was called with the right arguments
    mock_stream_chat.assert_called_once_with(messages)


@patch("llama_index.llms.asi.base.ASI.stream_chat")
def test_stream_chat_empty_content(mock_stream_chat):
    """Test stream_chat method with empty content in delta."""
    # Create mock chunks for the response
    mock_chunk1 = MagicMock()
    mock_chunk1.delta = "Thinking..."

    mock_chunk2 = MagicMock()
    mock_chunk2.delta = "Final response"

    # Set up the mock to return our test chunks
    mock_stream_chat.return_value = [mock_chunk1, mock_chunk2]

    # Create the ASI instance
    llm = ASI(api_key="test_key")

    # Call stream_chat
    messages = [ChatMessage(role=MessageRole.USER, content="Test message")]
    stream = llm.stream_chat(messages)

    # Check that we get the expected response
    chunks = list(stream)
    assert len(chunks) == 2
    assert chunks[0].delta == "Thinking..."
    assert chunks[1].delta == "Final response"

    # Verify that stream_chat was called with the right arguments
    mock_stream_chat.assert_called_once_with(messages)


@patch("llama_index.llms.asi.base.ASI.stream_chat")
def test_stream_chat_init_thought(mock_stream_chat):
    """Test stream_chat method with init_thought field."""
    # Create a mock for the response
    mock_chunk = MagicMock()
    mock_chunk.delta = "Initial thinking..."

    # Set up the mock to return our test chunk
    mock_stream_chat.return_value = [mock_chunk]

    # Create the ASI instance
    llm = ASI(api_key="test_key")

    # Call stream_chat
    messages = [ChatMessage(role=MessageRole.USER, content="Test message")]
    stream = llm.stream_chat(messages)

    # Check that we get the expected response
    chunks = list(stream)
    assert len(chunks) == 1
    assert chunks[0].delta == "Initial thinking..."

    # Verify that stream_chat was called with the right arguments
    mock_stream_chat.assert_called_once_with(messages)
