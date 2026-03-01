"""Tests for the SimpleSummarize response synthesizer."""

import pytest

from llama_index.core.llms import MockLLM
from llama_index.core.response_synthesizers.simple_summarize import SimpleSummarize


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_simple_summarize_default_construction() -> None:
    """SimpleSummarize should be constructable with default arguments."""
    synthesizer = SimpleSummarize(llm=MockLLM())
    assert synthesizer is not None


# ---------------------------------------------------------------------------
# get_response (sync)
# ---------------------------------------------------------------------------


def test_simple_summarize_get_response_non_streaming() -> None:
    """get_response should return a non-empty string for a non-streaming call."""
    synthesizer = SimpleSummarize(llm=MockLLM(), streaming=False)
    result = synthesizer.get_response(
        query_str="What is this?",
        text_chunks=["This is some context."],
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_simple_summarize_get_response_multiple_chunks() -> None:
    """get_response joins multiple chunks before summarising."""
    synthesizer = SimpleSummarize(llm=MockLLM(), streaming=False)
    result = synthesizer.get_response(
        query_str="Explain",
        text_chunks=["Chunk A.", "Chunk B.", "Chunk C."],
    )
    assert isinstance(result, str)


def test_simple_summarize_get_response_empty_chunks() -> None:
    """get_response should handle empty text_chunks without raising."""
    synthesizer = SimpleSummarize(llm=MockLLM(), streaming=False)
    result = synthesizer.get_response(query_str="test", text_chunks=[])
    # MockLLM returns the query as-is; result should be a string
    assert isinstance(result, str)


def test_simple_summarize_streaming_returns_generator() -> None:
    """When streaming=True, get_response should return a generator."""
    import types

    synthesizer = SimpleSummarize(llm=MockLLM(), streaming=True)
    result = synthesizer.get_response(
        query_str="test", text_chunks=["some text"]
    )
    # streaming response is a generator
    assert hasattr(result, "__iter__") or hasattr(result, "__next__")


# ---------------------------------------------------------------------------
# aget_response (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_summarize_aget_response_non_streaming() -> None:
    """aget_response should return a string in non-streaming mode."""
    synthesizer = SimpleSummarize(llm=MockLLM(), streaming=False)
    result = await synthesizer.aget_response(
        query_str="What is this?", text_chunks=["Context here."]
    )
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_simple_summarize_aget_response_multiple_chunks() -> None:
    """aget_response should join multiple chunks and return a summary."""
    synthesizer = SimpleSummarize(llm=MockLLM(), streaming=False)
    result = await synthesizer.aget_response(
        query_str="Summarize", text_chunks=["A.", "B.", "C."]
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_simple_summarize_aget_response_streaming() -> None:
    """aget_response with streaming=True should return an async generator."""
    synthesizer = SimpleSummarize(llm=MockLLM(), streaming=True)
    result = await synthesizer.aget_response(
        query_str="test", text_chunks=["text"]
    )
    assert hasattr(result, "__aiter__") or hasattr(result, "__iter__")


# ---------------------------------------------------------------------------
# synthesize (full pipeline)
# ---------------------------------------------------------------------------


def test_simple_summarize_synthesize() -> None:
    """synthesize() should return a Response wrapping the LLM output."""
    from llama_index.core.schema import NodeWithScore, TextNode

    synthesizer = SimpleSummarize(llm=MockLLM())
    node = NodeWithScore(node=TextNode(text="Relevant context."), score=1.0)
    response = synthesizer.synthesize(query="What is this about?", nodes=[node])
    assert response is not None
    assert str(response) != ""


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def test_simple_summarize_get_prompts() -> None:
    """_get_prompts should expose the text_qa_template."""
    synthesizer = SimpleSummarize(llm=MockLLM())
    prompts = synthesizer._get_prompts()
    assert "text_qa_template" in prompts


def test_simple_summarize_update_prompts() -> None:
    """_update_prompts should replace the qa template."""
    from llama_index.core.prompts.default_prompt_selectors import (
        DEFAULT_TEXT_QA_PROMPT_SEL,
    )

    synthesizer = SimpleSummarize(llm=MockLLM())
    synthesizer._update_prompts({"text_qa_template": DEFAULT_TEXT_QA_PROMPT_SEL})
    assert synthesizer._text_qa_template is DEFAULT_TEXT_QA_PROMPT_SEL


def test_simple_summarize_update_prompts_ignores_unknown_keys() -> None:
    """_update_prompts should not raise on unknown keys."""
    synthesizer = SimpleSummarize(llm=MockLLM())
    original = synthesizer._text_qa_template
    synthesizer._update_prompts({"unknown_key": "value"})
    assert synthesizer._text_qa_template is original
