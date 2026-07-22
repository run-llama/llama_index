import os

import pytest

from llama_index.llms.groq import Groq


@pytest.mark.skipif("GROQ_API_KEY" not in os.environ, reason="No Groq API key")
def test_completion():
    groq = Groq(model="openai/gpt-oss-120b", temperature=0, max_tokens=256)
    resp = groq.complete("hello")
    assert isinstance(resp.text, str)
    assert len(resp.text.strip()) > 0


@pytest.mark.skipif("GROQ_API_KEY" not in os.environ, reason="No Groq API key")
def test_stream_completion():
    groq = Groq(model="openai/gpt-oss-120b", temperature=0, max_tokens=256)
    stream = groq.stream_complete("hello")
    text = ""
    for chunk in stream:
        text = chunk.text
    assert len(text.strip()) > 0
