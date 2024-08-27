import os

import pytest

from llama_index.llms.cerebras import Cerebras


@pytest.mark.skipif("CEREBRAS_API_KEY" not in os.environ, reason="No Cerebras API key")
def test_completion():
    cerebras = Cerebras(model="llama3.1-8b")
    resp = cerebras.complete("What color is the sky? Answer in one word.")
    assert resp.text == "Blue."


@pytest.mark.skipif("CEREBRAS_API_KEY" not in os.environ, reason="No Cerebras API key")
def test_stream_completion():
    cerebras = Cerebras(model="llama3.1-8b")
    stream = cerebras.stream_complete(
        "What is 123 + 456? Embed the answer in the stories of the three little pigs"
    )
    text = ""
    for chunk in stream:
        text += chunk.delta
    assert "579" in text
