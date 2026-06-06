"""Regression tests for splitters crashing on an indivisible unit that is
larger than ``chunk_size`` -- e.g. a multi-token CJK / emoji character with a
small ``chunk_size``.

Before the fix, ``_split`` recursed on the same text forever (``RecursionError``)
and ``TokenTextSplitter._merge`` then popped from an empty list (``IndexError``).
"""

import logging

import pytest
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter


def test_token_splitter_oversized_unit_is_kept_not_recursed() -> None:
    # "🚀" encodes to 3 tokens; with chunk_size=1 it cannot be split further.
    # (A separate, pre-existing malformed warning in _merge is handled in #21796,
    # so silence that logger here to keep this test focused on the crash.)
    logging.getLogger("llama_index.core.node_parser.text.token").setLevel(
        logging.ERROR
    )
    splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
    assert splitter.split_text("🚀") == ["🚀"]
    assert splitter.split_text("🚀" * 20) == ["🚀"] * 20


def test_sentence_splitter_oversized_unit_raises_clean_error() -> None:
    # SentenceSplitter._merge already intends to raise this; the recursion
    # previously crashed before that path was ever reached.
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0)
    with pytest.raises(ValueError, match="Single token exceeded chunk size"):
        splitter.split_text("보험" * 50)


def test_splitters_unaffected_for_normal_text() -> None:
    text = "보험계약자는 보험료를 납입할 의무가 있다. 보험자는 보험금을 지급한다. " * 5
    assert len(SentenceSplitter(chunk_size=20, chunk_overlap=5).split_text(text)) > 1
    assert len(TokenTextSplitter(chunk_size=10, chunk_overlap=2).split_text(text)) > 1
