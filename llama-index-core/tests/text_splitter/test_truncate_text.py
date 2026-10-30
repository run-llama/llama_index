"""Regression tests for truncate_text() handling text that yields no chunks."""

import pytest

from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.node_parser.text.utils import truncate_text


@pytest.fixture()
def splitter() -> SentenceSplitter:
    return SentenceSplitter()


def test_truncate_text_empty_string_returns_empty(splitter: SentenceSplitter) -> None:
    """split_text('') returns no chunks, so truncate_text must not index [0]."""
    assert truncate_text("", splitter) == ""


def test_truncate_text_whitespace_only_returns_empty(splitter: SentenceSplitter) -> None:
    """Whitespace-only text also yields no chunks."""
    assert truncate_text("   ", splitter) == ""
    assert truncate_text("\n\n", splitter) == ""


def test_truncate_text_returns_first_chunk(splitter: SentenceSplitter) -> None:
    """Normal, oversized text is still truncated to the first chunk."""
    text = "Sentence one. " * 200
    result = truncate_text(text, splitter)
    assert result
    assert result.startswith("Sentence one.")
    assert len(result) < len(text)
