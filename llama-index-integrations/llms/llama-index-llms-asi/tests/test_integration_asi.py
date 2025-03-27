import os
import json

import pytest
from llama_index.llms.openai.base import ChatMessage, MessageRole

# Import ASI directly from the local package
from llama_index.llms.asi.base import ASI


@pytest.mark.skipif("ASI_API_KEY" not in os.environ, reason="No ASI API key")
def test_completion():
    asi = ASI(model="asi1-mini", temperature=0, max_tokens=10)
    resp = asi.complete("hello")
    assert resp.text.strip() != ""


@pytest.mark.skipif("ASI_API_KEY" not in os.environ, reason="No ASI API key")
def test_chat():
    """Test chat functionality."""
    asi = ASI(model="asi1-mini", temperature=0, max_tokens=10)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful AI assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?"),
    ]
    response = asi.chat(messages)
    assert response.message.content.strip() != ""


@pytest.mark.skipif("ASI_API_KEY" not in os.environ, reason="No ASI API key")
def test_stream_completion():
    """Test streaming completion functionality."""
    asi = ASI(model="asi1-mini", temperature=0, max_tokens=10)
    stream = asi.stream_complete("hello")
    text = None
    for chunk in stream:
        if chunk.text:
            text = chunk.text
    
    # With our custom handler, we should get a non-empty response
    assert text is not None
    assert text.strip() != ""


@pytest.mark.skipif("ASI_API_KEY" not in os.environ, reason="No ASI API key")
def test_stream_chat():
    """Test streaming chat functionality with custom handler."""
    asi = ASI(model="asi1-mini", temperature=0, max_tokens=10)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful AI assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?"),
    ]
    stream = asi.stream_chat(messages)
    content = None
    for chunk in stream:
        if chunk.message and chunk.message.content:
            content = chunk.message.content
    
    # With our custom handler, we should get a non-empty response
    assert content is not None
    assert content.strip() != ""
