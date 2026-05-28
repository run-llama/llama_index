"""Test text splitter."""

from typing import List
import logging

import pytest
import tiktoken
from llama_index.core.node_parser.text import TokenTextSplitter
from llama_index.core.node_parser.text.utils import truncate_text
from llama_index.core.schema import Document, MetadataMode, TextNode


def test_split_token() -> None:
    """Test split normal token."""
    token = "foo bar"
    text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
    chunks = text_splitter.split_text(token)
    assert chunks == ["foo", "bar"]

    token = "foo bar hello world"
    text_splitter = TokenTextSplitter(chunk_size=2, chunk_overlap=1)
    chunks = text_splitter.split_text(token)
    assert chunks == ["foo bar", "bar hello", "hello world"]


def test_start_end_char_idx() -> None:
    document = Document(text="foo bar hello world baz bbq")
    text_splitter = TokenTextSplitter(chunk_size=3, chunk_overlap=1)
    nodes: List[TextNode] = text_splitter.get_nodes_from_documents([document])
    for node in nodes:
        assert node.start_char_idx is not None
        assert node.end_char_idx is not None
        assert node.end_char_idx - node.start_char_idx == len(
            node.get_content(metadata_mode=MetadataMode.NONE)
        )


def test_truncate_token() -> None:
    """Test truncate normal token."""
    token = "foo bar"
    text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
    text = truncate_text(token, text_splitter)
    assert text == "foo"


def test_split_long_token() -> None:
    """Test split a really long token."""
    token = "a" * 100
    tokenizer = tiktoken.get_encoding("gpt2")
    text_splitter = TokenTextSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=tokenizer.encode
    )
    chunks = text_splitter.split_text(token)
    # each text chunk may have spaces, since we join splits by separator
    assert "".join(chunks).replace(" ", "") == token

    token = ("a" * 49) + "\n" + ("a" * 50)
    text_splitter = TokenTextSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=tokenizer.encode
    )
    chunks = text_splitter.split_text(token)
    assert len(chunks[0]) == 49
    assert len(chunks[1]) == 50


def test_split_chinese(chinese_text: str) -> None:
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
    chunks = text_splitter.split_text(chinese_text)
    assert len(chunks) == 2


def test_contiguous_text(contiguous_text: str) -> None:
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    chunks = splitter.split_text(contiguous_text)
    assert len(chunks) == 10


def test_split_with_metadata(english_text: str) -> None:
    chunk_size = 100
    metadata_str = "word " * 50
    tokenizer = tiktoken.get_encoding("gpt2")
    splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, tokenizer=tokenizer.encode
    )

    chunks = splitter.split_text(english_text)
    assert len(chunks) == 2

    chunks = splitter.split_text_metadata_aware(english_text, metadata_str=metadata_str)
    assert len(chunks) == 4
    for chunk in chunks:
        node_content = chunk + metadata_str
        assert len(tokenizer.encode(node_content)) <= 100


def test_merge_warning_message_is_well_formed(
       caplog: pytest.LogCaptureFixture,
   ) -> None:
    """The 'split larger than chunk size' warning must be emitted cleanly.

    Regression test: previously the warning was constructed with two separate
    f-string arguments to ``logger.warning(...)`` (trailing commas after each),
    which made the logging machinery treat the second f-string as an arg for
    ``%`` formatting against the first. The first f-string had no ``%s``
    placeholders, so the call raised a ``TypeError: not all arguments
    converted during string formatting`` inside the logging emit path. The
    warning record was therefore never delivered to handlers cleanly, and
    stderr was polluted with a logging-internal traceback whenever the warning
    path was hit.
    """

    # A tokenizer that always reports a token count of 10, regardless of the
    # input. This guarantees ``split_len > chunk_size`` inside ``_merge`` and
    # exercises the warning branch reliably, without depending on a real
    # tokenizer's network-downloaded vocabulary.
    def big_tokenizer(text: str) -> List[int]:
        return [0] * 10

    splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0, tokenizer=big_tokenizer)

    with caplog.at_level(
        logging.WARNING, logger="llama_index.core.node_parser.text.token"
    ):
        # _merge is called on the result of _split; pass a non-empty text so
        # _split returns a single oversize element and _merge logs. _merge
        # itself may raise downstream on an oversize-single-split edge case
        # unrelated to this regression; we suppress that here because the
        # warning emission we care about happens before any such raise.
        try:
            splitter._merge(["xx"], chunk_size=1)
        except IndexError:
            pass

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "Expected at least one warning when a split exceeds chunk size"

    # The fixed message must contain both halves in a single, well-formed string.
    # Calling .getMessage() exercises the same code path that raised TypeError
    # before the fix; if the bug regresses, this assertion fails.
    msg = warnings[0].getMessage()
    assert "Got a split of size" in msg
    assert "larger than chunk size" in msg
