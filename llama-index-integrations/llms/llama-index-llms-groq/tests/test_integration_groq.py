import os

import pytest

from llama_index.llms.groq import Groq


@pytest.mark.skipif("GROQ_API_KEY" not in os.environ, reason="No Groq API key")
def test_completion():
    groq = Groq(model="mixtral-8x7b-32768", temperature=0, max_tokens=2)
    resp = groq.complete("hello")
    assert resp.text == "Hello"


@pytest.mark.skipif("GROQ_API_KEY" not in os.environ, reason="No Groq API key")
def test_stream_completion():
    groq = Groq(model="mixtral-8x7b-32768", temperature=0, max_tokens=2)
    stream = groq.stream_complete("hello")
    text = None
    for chunk in stream:
        text = chunk.text
    assert text == "Hello"
