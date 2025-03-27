import os
import pytest

from llama_index.llms.asi import ASI
from llama_index.llms.openai.base import ChatMessage, MessageRole


@pytest.mark.skipif("ASI_API_KEY" not in os.environ, reason="No ASI API key")
def test_completion():
    asi = ASI(model="asi1-mini", temperature=0, max_tokens=10)
    resp = asi.complete("hello")
    assert resp.text.strip() != ""


@pytest.mark.skipif("ASI_API_KEY" not in os.environ, reason="No ASI API key")
def test_chat():
    asi = ASI(model="asi1-mini", temperature=0, max_tokens=10)
    messages = [
        ChatMessage(role=MessageRole.USER, content="hello")
    ]
    resp = asi.chat(messages)
    assert resp.message.content.strip() != ""


@pytest.mark.skipif("ASI_API_KEY" not in os.environ, reason="No ASI API key")
def test_stream_completion():
    asi = ASI(model="asi1-mini", temperature=0, max_tokens=10)
    resp_gen = asi.stream_complete("hello")

    # Collect all chunks
    chunks = []
    for chunk in resp_gen:
        chunks.append(chunk)

    # Verify we got at least one chunk with content
    assert len(chunks) > 0
    assert chunks[0].text.strip() != ""


@pytest.mark.skipif("ASI_API_KEY" not in os.environ, reason="No ASI API key")
def test_stream_chat():
    asi = ASI(model="asi1-mini", temperature=0, max_tokens=10)
    messages = [
        ChatMessage(role=MessageRole.USER, content="hello")
    ]
    resp_gen = asi.stream_chat(messages)

    # Collect all chunks
    chunks = []
    for chunk in resp_gen:
        chunks.append(chunk)

    # Verify we got at least one chunk with content
    assert len(chunks) > 0
