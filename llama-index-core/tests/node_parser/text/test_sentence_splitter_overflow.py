"""Tests for SentenceSplitter chunk overflow after whitespace stripping.

Regression tests for https://github.com/run-llama/llama_index/issues/23575
"""

from typing import List

import pytest

from llama_index.core.node_parser.text.sentence import SentenceSplitter


def _char_tokenizer(text: str) -> List[int]:
    """1 token = 1 character, for deterministic tests."""
    return list(range(len(text)))


class TestChunkSizeAfterStripping:
    """Every emitted chunk must be <= chunk_size tokens, even after stripping."""

    def test_trailing_whitespace_does_not_overflow(self):
        """Chunk that is exactly chunk_size before stripping must not exceed
        chunk_size after stripping changes tokenization."""
        splitter = SentenceSplitter(
            chunk_size=10,
            chunk_overlap=0,
            tokenizer=_char_tokenizer,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^.]+\\.?",
        )
        # 10 chars with trailing spaces — fits in chunk_size=10
        # After stripping, BPE re-tokenization could push it over.
        # With char tokenizer: "aaaaaaaaaa  " (12) -> stripped "aaaaaaaaaa" (10) = ok
        text = "aaa.  bbb.  ccc.  ddd."
        chunks = splitter.split_text(text)
        for chunk in chunks:
            token_count = len(_char_tokenizer(chunk))
            assert token_count <= 10, (
                f"Chunk exceeds chunk_size after stripping: "
                f"{token_count} tokens in {chunk!r}"
            )

    def test_all_chunks_within_limit(self):
        """No chunk should ever exceed chunk_size, regardless of content."""
        splitter = SentenceSplitter(
            chunk_size=8,
            chunk_overlap=0,
            tokenizer=_char_tokenizer,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^.]+\\.?",
        )
        text = "Hello.  World.  Foo.  Bar.  Baz."
        chunks = splitter.split_text(text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(_char_tokenizer(chunk)) <= 8

    def test_whitespace_only_chunks_removed(self):
        """Chunks that become empty after stripping should be removed."""
        splitter = SentenceSplitter(
            chunk_size=50,
            chunk_overlap=0,
            tokenizer=_char_tokenizer,
        )
        text = "Hello world."
        chunks = splitter.split_text(text)
        for chunk in chunks:
            assert chunk.strip() != "", "Empty chunk should have been removed"

    def test_emoji_boundary_no_overflow(self):
        """Emoji + whitespace boundaries are a known BPE edge case."""
        splitter = SentenceSplitter(
            chunk_size=5,
            chunk_overlap=0,
            tokenizer=_char_tokenizer,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^.]+\\.?",
        )
        text = "😀. 😀. 😀. 😀."
        chunks = splitter.split_text(text)
        for chunk in chunks:
            assert len(_char_tokenizer(chunk)) <= 5

    def test_truncate_preserves_content(self):
        """Truncation should keep as much content as possible."""
        splitter = SentenceSplitter(
            chunk_size=5,
            chunk_overlap=0,
            tokenizer=_char_tokenizer,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^.]+\\.?",
        )
        text = "ab. cd. ef."
        chunks = splitter.split_text(text)
        joined = " ".join(chunks)
        # All original non-whitespace content should be present across chunks
        for char in "abcdef":
            assert char in joined, f"Character {char!r} lost during splitting"
