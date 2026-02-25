"""Tests for the Accumulate response synthesizer."""

import pytest

from llama_index.core.llms import MockLLM
from llama_index.core.response_synthesizers.accumulate import Accumulate


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_accumulate_default_construction() -> None:
    """Accumulate should be constructable with default arguments."""
    synthesizer = Accumulate(llm=MockLLM())
    assert synthesizer is not None


def test_accumulate_streaming_raises_on_get_response() -> None:
    """Accumulate does not support streaming; get_response should raise."""
    synthesizer = Accumulate(llm=MockLLM(), streaming=True)
    with pytest.raises(ValueError, match="Unable to stream"):
        synthesizer.get_response(query_str="test", text_chunks=["chunk1"])


@pytest.mark.asyncio
async def test_accumulate_streaming_raises_on_aget_response() -> None:
    """aget_response should also raise when streaming=True."""
    synthesizer = Accumulate(llm=MockLLM(), streaming=True)
    with pytest.raises(ValueError, match="Unable to stream"):
        await synthesizer.aget_response(query_str="test", text_chunks=["chunk1"])


# ---------------------------------------------------------------------------
# get_response
# ---------------------------------------------------------------------------


def test_accumulate_single_chunk() -> None:
    """A single chunk should produce one numbered response."""
    synthesizer = Accumulate(llm=MockLLM())
    result = synthesizer.get_response(
        query_str="What is the capital?", text_chunks=["Paris is the capital."]
    )
    assert isinstance(result, str)
    assert "Response 1:" in result


def test_accumulate_multiple_chunks_format() -> None:
    """Multiple chunks produce multiple numbered responses separated by the separator."""
    synthesizer = Accumulate(llm=MockLLM())
    chunks = ["Chunk A", "Chunk B", "Chunk C"]
    sep = "\n-----\n"
    result = synthesizer.get_response(
        query_str="Summarize", text_chunks=chunks, separator=sep
    )
    assert "Response 1:" in result
    assert "Response 2:" in result
    assert "Response 3:" in result
    assert sep in result


def test_accumulate_empty_chunks() -> None:
    """Empty text_chunks should return an empty formatted string."""
    synthesizer = Accumulate(llm=MockLLM())
    result = synthesizer.get_response(query_str="test", text_chunks=[])
    # No responses to accumulate; result should be empty or have no "Response N:"
    assert "Response 1:" not in result


def test_accumulate_custom_separator() -> None:
    """The separator kwarg should be used to join responses."""
    synthesizer = Accumulate(llm=MockLLM())
    custom_sep = " | "
    result = synthesizer.get_response(
        query_str="test",
        text_chunks=["a", "b"],
        separator=custom_sep,
    )
    assert custom_sep in result


# ---------------------------------------------------------------------------
# aget_response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accumulate_aget_single_chunk() -> None:
    """aget_response with one chunk should produce a valid string."""
    synthesizer = Accumulate(llm=MockLLM())
    result = await synthesizer.aget_response(
        query_str="What is 1+1?", text_chunks=["2"]
    )
    assert isinstance(result, str)
    assert "Response 1:" in result


@pytest.mark.asyncio
async def test_accumulate_aget_multiple_chunks() -> None:
    """aget_response with multiple chunks returns all numbered responses."""
    synthesizer = Accumulate(llm=MockLLM())
    chunks = ["X", "Y"]
    result = await synthesizer.aget_response(query_str="test", text_chunks=chunks)
    assert "Response 1:" in result
    assert "Response 2:" in result


# ---------------------------------------------------------------------------
# synthesize (full pipeline)
# ---------------------------------------------------------------------------


def test_accumulate_synthesize_returns_response() -> None:
    """synthesize() should return a Response object."""
    from llama_index.core.schema import NodeWithScore, TextNode

    synthesizer = Accumulate(llm=MockLLM())
    node = NodeWithScore(node=TextNode(text="Some context."), score=1.0)
    response = synthesizer.synthesize(query="What is this?", nodes=[node])
    assert response is not None
    assert str(response) != ""


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def test_accumulate_get_prompts() -> None:
    """_get_prompts should return the text_qa_template key."""
    synthesizer = Accumulate(llm=MockLLM())
    prompts = synthesizer._get_prompts()
    assert "text_qa_template" in prompts


def test_accumulate_update_prompts() -> None:
    """_update_prompts should replace the text_qa_template."""
    from llama_index.core.prompts.default_prompt_selectors import (
        DEFAULT_TEXT_QA_PROMPT_SEL,
    )

    synthesizer = Accumulate(llm=MockLLM())
    original_template = synthesizer._text_qa_template

    synthesizer._update_prompts({"text_qa_template": DEFAULT_TEXT_QA_PROMPT_SEL})
    assert synthesizer._text_qa_template is DEFAULT_TEXT_QA_PROMPT_SEL


def test_accumulate_flatten_list() -> None:
    """flatten_list should collapse a list-of-lists into a flat list."""
    synthesizer = Accumulate(llm=MockLLM())
    nested = [[1, 2], [3], [4, 5]]
    assert synthesizer.flatten_list(nested) == [1, 2, 3, 4, 5]
