from unittest.mock import MagicMock, patch
import pytest
import os

from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.asi import ASI
from llama_index.llms.openai.base import ChatMessage, MessageRole, CompletionResponse, ChatResponse


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
    with pytest.raises(ValueError, match="ASI API key is required"):
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
    mock_chunk.raw.choices = [MagicMock()]
    mock_chunk.raw.choices[0].delta.content = "Test content"
    mock_chunk.message = None
    
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
    assert chunks[0].message.content == "Test content"
    assert chunks[0].message.role == MessageRole.ASSISTANT


@patch("llama_index.llms.openai_like.OpenAILike.stream_chat")
@patch("llama_index.llms.asi.base.ASI.chat")
def test_stream_chat_empty_content(mock_chat, mock_stream_chat):
    """Test stream_chat method with empty content in delta."""
    # Create a mock for the raw stream with empty content
    mock_chunk1 = MagicMock()
    mock_chunk1.raw.choices = [MagicMock()]
    mock_chunk1.raw.choices[0].delta.content = ""
    mock_chunk1.message = None
    
    # Create a final chunk with usage info
    mock_chunk2 = MagicMock()
    mock_chunk2.additional_kwargs = {"prompt_tokens": 10}
    mock_chunk2.raw = MagicMock()
    
    # Set up the stream_chat mock to return our test chunks
    mock_stream_chat.return_value = [mock_chunk1, mock_chunk2]
    
    # Set up the chat mock to prevent actual API calls
    mock_response = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="Fallback response")
    )
    mock_chat.return_value = mock_response
    
    # Create the ASI instance
    llm = ASI(api_key="test_key")
    
    # Call stream_chat
    messages = [ChatMessage(role=MessageRole.USER, content="Test message")]
    stream = llm.stream_chat(messages)
    
    # We should get the fallback response since all content was empty
    chunks = list(stream)
    assert len(chunks) == 1
    assert chunks[0].message.content == "Fallback response"


@patch("llama_index.llms.openai_like.OpenAILike.stream_chat")
@patch("llama_index.llms.openai_like.OpenAILike.chat")
def test_stream_chat_fallback(mock_chat, mock_stream_chat):
    """Test stream_chat method falling back to chat."""
    # Create a mock for the raw stream with empty content
    mock_chunk1 = MagicMock()
    mock_chunk1.raw.choices = [MagicMock()]
    mock_chunk1.raw.choices[0].delta.content = ""
    mock_chunk1.message = None
    
    # Create a final chunk with usage info
    mock_chunk2 = MagicMock()
    mock_chunk2.additional_kwargs = {"prompt_tokens": 10}
    mock_chunk2.raw = MagicMock()
    
    # Set up the stream_chat mock to return our test chunks
    mock_stream_chat.return_value = [mock_chunk1, mock_chunk2]
    
    # Set up the chat mock
    mock_response = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="Fallback response")
    )
    mock_chat.return_value = mock_response
    
    # Create the ASI instance
    llm = ASI(api_key="test_key")
    
    # Call stream_chat
    messages = [ChatMessage(role=MessageRole.USER, content="Test message")]
    stream = llm.stream_chat(messages)
    
    # Check that we get the fallback response
    chunks = list(stream)
    assert len(chunks) == 1
    assert chunks[0].message.content == "Fallback response"
    
    # Verify that chat was called with the right arguments
    mock_chat.assert_called_once_with(messages)
