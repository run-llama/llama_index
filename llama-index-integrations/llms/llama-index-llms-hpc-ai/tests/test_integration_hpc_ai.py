import os

import pytest

from llama_index.llms.hpc_ai import HpcAiLLM


@pytest.mark.skipif("HPC_AI_API_KEY" not in os.environ, reason="No HPC-AI API key")
def test_completion():
    llm = HpcAiLLM(model="minimax/minimax-m2.5", temperature=0, max_tokens=32)
    resp = llm.complete("Say only the word hello in lowercase.")
    assert resp.text.strip()


@pytest.mark.skipif("HPC_AI_API_KEY" not in os.environ, reason="No HPC-AI API key")
def test_stream_completion():
    llm = HpcAiLLM(model="minimax/minimax-m2.5", temperature=0, max_tokens=32)
    stream = llm.stream_complete("Say only the word hello in lowercase.")
    text = None
    for chunk in stream:
        text = chunk.text
    assert text is not None
    assert text.strip()
