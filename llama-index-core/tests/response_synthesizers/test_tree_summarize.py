"""Tests for the TreeSummarize response synthesizer."""

import pytest

from llama_index.core.llms import MockLLM
from llama_index.core.response_synthesizers.tree_summarize import TreeSummarize


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_tree_summarize_default_construction() -> None:
    """TreeSummarize should be constructable with only an LLM."""
    synthesizer = TreeSummarize(llm=MockLLM())
    assert synthesizer is not None


def test_tree_summarize_verbose_flag() -> None:
    """verbose flag should be stored on the instance."""
    synthesizer = TreeSummarize(llm=MockLLM(), verbose=True)
    assert synthesizer._verbose is True


def test_tree_summarize_use_async_flag() -> None:
    """use_async flag should be stored on the instance."""
    synthesizer = TreeSummarize(llm=MockLLM(), use_async=True)
    assert synthesizer._use_async is True


# ---------------------------------------------------------------------------
# get_response (sync via aget_response internally)
# ---------------------------------------------------------------------------


def test_tree_summarize_synthesize_single_chunk() -> None:
    """synthesize() with a single node should return a non-empty response."""
    from llama_index.core.schema import NodeWithScore, TextNode

    synthesizer = TreeSummarize(llm=MockLLM())
    node = NodeWithScore(node=TextNode(text="Paris is the capital of France."), score=1.0)
    response = synthesizer.synthesize(query="What is the capital of France?", nodes=[node])
    assert response is not None
    assert str(response) != ""


def test_tree_summarize_synthesize_multiple_nodes() -> None:
    """synthesize() with multiple nodes should still return a valid response."""
    from llama_index.core.schema import NodeWithScore, TextNode

    synthesizer = TreeSummarize(llm=MockLLM())
    nodes = [
        NodeWithScore(node=TextNode(text=f"Chunk {i}."), score=float(i))
        for i in range(3)
    ]
    response = synthesizer.synthesize(query="Summarize all chunks.", nodes=nodes)
    assert response is not None


def test_tree_summarize_synthesize_no_nodes() -> None:
    """synthesize() with empty nodes should not raise."""
    synthesizer = TreeSummarize(llm=MockLLM())
    response = synthesizer.synthesize(query="test", nodes=[])
    assert response is not None


# ---------------------------------------------------------------------------
# aget_response (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tree_summarize_aget_response_single_chunk() -> None:
    """aget_response should return a string for a single text chunk."""
    synthesizer = TreeSummarize(llm=MockLLM())
    result = await synthesizer.aget_response(
        query_str="test question", text_chunks=["Single chunk of text."]
    )
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_tree_summarize_aget_response_multiple_chunks() -> None:
    """aget_response with multiple chunks should recurse and produce a string."""
    synthesizer = TreeSummarize(llm=MockLLM())
    chunks = [f"Chunk number {i} with content." for i in range(4)]
    result = await synthesizer.aget_response(
        query_str="Summarize", text_chunks=chunks
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_tree_summarize_aget_response_empty_chunks() -> None:
    """aget_response with empty chunks should return an empty string or handle gracefully."""
    synthesizer = TreeSummarize(llm=MockLLM())
    # Empty list - behavior depends on prompt_helper.repack; should not raise
    try:
        result = await synthesizer.aget_response(
            query_str="test", text_chunks=[]
        )
        assert isinstance(result, str)
    except (ValueError, IndexError):
        # Acceptable to raise on truly empty input after repacking
        pass


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def test_tree_summarize_get_prompts() -> None:
    """_get_prompts should expose the summary_template key."""
    synthesizer = TreeSummarize(llm=MockLLM())
    prompts = synthesizer._get_prompts()
    assert "summary_template" in prompts


def test_tree_summarize_update_prompts() -> None:
    """_update_prompts should allow replacing the summary template."""
    from llama_index.core.prompts.default_prompt_selectors import (
        DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
    )

    synthesizer = TreeSummarize(llm=MockLLM())
    synthesizer._update_prompts({"summary_template": DEFAULT_TREE_SUMMARIZE_PROMPT_SEL})
    assert synthesizer._summary_template is DEFAULT_TREE_SUMMARIZE_PROMPT_SEL


def test_tree_summarize_update_prompts_ignores_unknown() -> None:
    """_update_prompts should not raise on unknown keys."""
    synthesizer = TreeSummarize(llm=MockLLM())
    original = synthesizer._summary_template
    synthesizer._update_prompts({"unknown": "value"})
    assert synthesizer._summary_template is original


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_tree_summarize_streaming_construction() -> None:
    """TreeSummarize with streaming=True should construct without error."""
    synthesizer = TreeSummarize(llm=MockLLM(), streaming=True)
    assert synthesizer is not None


@pytest.mark.asyncio
async def test_tree_summarize_aget_response_streaming_single_chunk() -> None:
    """With streaming=True and a single chunk, aget_response returns a generator."""
    synthesizer = TreeSummarize(llm=MockLLM(), streaming=True)
    result = await synthesizer.aget_response(
        query_str="test", text_chunks=["Single chunk"]
    )
    # MockLLM's astream returns something iterable
    assert result is not None
