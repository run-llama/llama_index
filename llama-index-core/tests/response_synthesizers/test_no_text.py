"""Tests for the NoText response synthesizer."""

import pytest

from llama_index.core.response_synthesizers.no_text import NoText


def test_no_text_construction() -> None:
    """NoText should be constructable without arguments."""
    synthesizer = NoText()
    assert synthesizer is not None


def test_no_text_get_response_returns_empty_string() -> None:
    """get_response should always return an empty string."""
    synthesizer = NoText()
    result = synthesizer.get_response(
        query_str="Any query", text_chunks=["chunk1", "chunk2"]
    )
    assert result == ""


def test_no_text_get_response_empty_chunks() -> None:
    """get_response with empty chunks should still return empty string."""
    synthesizer = NoText()
    result = synthesizer.get_response(query_str="test", text_chunks=[])
    assert result == ""


@pytest.mark.asyncio
async def test_no_text_aget_response_returns_empty_string() -> None:
    """aget_response should always return an empty string."""
    synthesizer = NoText()
    result = await synthesizer.aget_response(
        query_str="Any query", text_chunks=["chunk1"]
    )
    assert result == ""


@pytest.mark.asyncio
async def test_no_text_aget_response_empty_chunks() -> None:
    """aget_response with no chunks should return empty string."""
    synthesizer = NoText()
    result = await synthesizer.aget_response(query_str="test", text_chunks=[])
    assert result == ""


def test_no_text_get_prompts_empty() -> None:
    """_get_prompts should return an empty dict (no templates needed)."""
    synthesizer = NoText()
    prompts = synthesizer._get_prompts()
    assert prompts == {}


def test_no_text_update_prompts_noop() -> None:
    """_update_prompts should not raise even with arbitrary input."""
    synthesizer = NoText()
    synthesizer._update_prompts({"any_key": "any_value"})


def test_no_text_synthesize_returns_empty_response() -> None:
    """synthesize() should return a Response with empty string content."""
    from llama_index.core.schema import NodeWithScore, TextNode

    synthesizer = NoText()
    node = NodeWithScore(node=TextNode(text="Some text"), score=1.0)
    response = synthesizer.synthesize(query="test", nodes=[node])
    assert str(response) == ""
