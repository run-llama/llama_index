import os
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.openai.base import (
    ChatMessage, CompletionResponse, MessageRole
)

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
        match="Must specify `api_key` or set environment variable "
              "`ASI_API_KEY`"
    ):
        ASI(api_key=None)


@patch("llama_index.llms.openai_like.OpenAILike.complete")
def test_stream_complete(mock_complete):
    """Test stream_complete method which should fall back to complete."""
    # Set up the mock
    mock_response = CompletionResponse(text="Test response")
    mock_complete.return_value = mock_response

    # Create the ASI instance
    llm = ASI(api_key="test_key")

    # Call stream_complete
    stream = llm.stream_complete("Test prompt")

    # Check that we get the expected response
    chunks = list(stream)
    assert len(chunks) == 1
    assert chunks[0].text == "Test response"

    # Verify that complete was called with the right arguments
    mock_complete.assert_called_once_with("Test prompt", formatted=False)


@patch("llama_index.llms.openai_like.OpenAILike.stream_chat")
def test_stream_chat_with_content(mock_stream_chat):
    """Test stream_chat method with content in delta."""
    # Create a mock for the raw stream
    mock_chunk = MagicMock()
    mock_chunk.delta = MagicMock()
    mock_chunk.delta.content = "Test content"
    mock_chunk.raw = {}

    # Set up the mock to return our test chunk
    mock_stream_chat.return_value = [mock_chunk]

    # Create the ASI instance
    llm = ASI(api_key="test_key")

    # Call stream_chat
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="Test message"
        )
    ]
    stream = llm.stream_chat(messages)

    # Check that we get the expected response
    chunks = list(stream)
    assert len(chunks) == 1


@patch("llama_index.llms.openai_like.OpenAILike.stream_chat")
def test_stream_chat_empty_content(mock_stream_chat):
    """Test stream_chat method with empty content in delta."""
    # Create a mock for the raw stream with empty content
    mock_chunk1 = MagicMock()
    mock_chunk1.delta = MagicMock()
    mock_chunk1.delta.content = None
    mock_chunk1.raw = {"thought": "Thinking..."}

    # Create a final chunk with content
    mock_chunk2 = MagicMock()
    mock_chunk2.delta = MagicMock()
    mock_chunk2.delta.content = "Final response"
    mock_chunk2.raw = {}

    # Set up the stream_chat mock to return our test chunks
    mock_stream_chat.return_value = [mock_chunk1, mock_chunk2]

    # Create the ASI instance
    llm = ASI(api_key="test_key")

    # Call stream_chat
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="Test message"
        )
    ]
    stream = llm.stream_chat(messages)

    # Get chunks, with the first one having content from the thought field
    chunks = list(stream)
    assert len(chunks) == 2
    assert chunks[0].delta.content == "Thinking..."
    assert chunks[1].delta.content == "Final response"


@patch("llama_index.llms.openai_like.OpenAILike.stream_chat")
def test_stream_chat_init_thought(mock_stream_chat):
    """Test stream_chat method with init_thought field."""
    # Create a mock for the raw stream with init_thought
    mock_chunk = MagicMock()
    mock_chunk.delta = MagicMock()
    mock_chunk.delta.content = None
    mock_chunk.raw = {
        "init_thought": "Initial thinking..."
    }

    # Set up the stream_chat mock to return our test chunk
    mock_stream_chat.return_value = [mock_chunk]

    # Create the ASI instance
    llm = ASI(api_key="test_key")

    # Call stream_chat
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="Test message"
        )
    ]
    stream = llm.stream_chat(messages)

    # We should get the chunk with content from the init_thought field
    chunks = list(stream)
    assert len(chunks) == 1
    assert chunks[0].delta.content == "Initial thinking..."
