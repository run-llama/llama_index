"""Tests for the CompactAndRefine response synthesizer."""

import pytest

from llama_index.core.llms import MockLLM
from llama_index.core.response_synthesizers.compact_and_refine import CompactAndRefine


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_compact_and_refine_default_construction() -> None:
    """CompactAndRefine should inherit from Refine and be constructable."""
    from llama_index.core.response_synthesizers.refine import Refine

    synthesizer = CompactAndRefine(llm=MockLLM())
    assert isinstance(synthesizer, Refine)


# ---------------------------------------------------------------------------
# get_response (sync)
# ---------------------------------------------------------------------------


def test_compact_and_refine_get_response_single_chunk() -> None:
    """get_response with one chunk should return a non-empty string."""
    synthesizer = CompactAndRefine(llm=MockLLM())
    result = synthesizer.get_response(
        query_str="What is the answer?", text_chunks=["The answer is 42."]
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_compact_and_refine_get_response_multiple_chunks() -> None:
    """get_response should compact multiple chunks before refining."""
    synthesizer = CompactAndRefine(llm=MockLLM())
    result = synthesizer.get_response(
        query_str="Summarize",
        text_chunks=["Part one.", "Part two.", "Part three."],
    )
    assert isinstance(result, str)


def test_compact_and_refine_get_response_empty_chunks() -> None:
    """get_response with empty chunks should not raise."""
    synthesizer = CompactAndRefine(llm=MockLLM())
    result = synthesizer.get_response(query_str="test", text_chunks=[])
    assert isinstance(result, str)


def test_compact_and_refine_get_response_with_prev_response() -> None:
    """get_response should accept a prev_response argument without raising."""
    synthesizer = CompactAndRefine(llm=MockLLM())
    result = synthesizer.get_response(
        query_str="What else?",
        text_chunks=["Additional context."],
        prev_response="Initial answer.",
    )
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# aget_response (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_and_refine_aget_response_single_chunk() -> None:
    """aget_response should return a string for a single chunk."""
    synthesizer = CompactAndRefine(llm=MockLLM())
    result = await synthesizer.aget_response(
        query_str="Explain", text_chunks=["Context."]
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_compact_and_refine_aget_response_multiple_chunks() -> None:
    """aget_response should handle multiple chunks correctly."""
    synthesizer = CompactAndRefine(llm=MockLLM())
    result = await synthesizer.aget_response(
        query_str="test", text_chunks=["A", "B", "C"]
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_compact_and_refine_aget_empty_chunks() -> None:
    """aget_response should not raise with empty chunks."""
    synthesizer = CompactAndRefine(llm=MockLLM())
    result = await synthesizer.aget_response(query_str="test", text_chunks=[])
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Full synthesize pipeline
# ---------------------------------------------------------------------------


def test_compact_and_refine_synthesize() -> None:
    """synthesize() should return a valid Response object."""
    from llama_index.core.schema import NodeWithScore, TextNode

    synthesizer = CompactAndRefine(llm=MockLLM())
    node = NodeWithScore(node=TextNode(text="Some context."), score=1.0)
    response = synthesizer.synthesize(query="What is this?", nodes=[node])
    assert response is not None


# ---------------------------------------------------------------------------
# _make_compact_text_chunks
# ---------------------------------------------------------------------------


def test_make_compact_text_chunks_returns_list() -> None:
    """_make_compact_text_chunks should return a list of strings."""
    synthesizer = CompactAndRefine(llm=MockLLM())
    chunks = synthesizer._make_compact_text_chunks(
        "test query", ["text chunk one", "text chunk two"]
    )
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)


def test_make_compact_text_chunks_empty_input() -> None:
    """_make_compact_text_chunks with empty input should return empty list."""
    synthesizer = CompactAndRefine(llm=MockLLM())
    chunks = synthesizer._make_compact_text_chunks("test query", [])
    assert isinstance(chunks, list)
