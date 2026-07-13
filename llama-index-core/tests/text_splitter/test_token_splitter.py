"""Test text splitter."""

from typing import List

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


@pytest.mark.parametrize(
    ("text", "chunk_size", "chunk_overlap"),
    [
        # original repro: a single word's fused leading-space token count
        # jumps once stripped, pushing the chunk to 21 tokens on unfixed code.
        ("coz rdburacy hfnppgmb mam zzoj wxzrv pegjgbs xkjqz", 20, 5),
        # same class of bug at a much tighter chunk_size, unrelated text.
        ("tqbmbylk npgzrka dmfjy vwtogkt tqgwioqr tacgtm xvaidhlqxq", 8, 0),
        # and again at a mid-range chunk_size.
        ("hoytyof gjzsrwahyf hvbzlrk dodgzcbn nwdfto", 12, 0),
    ],
)
def test_merge_respects_chunk_size_across_whitespace_strip(
    text: str, chunk_size: int, chunk_overlap: int
) -> None:
    """Regression test: emitted chunks must never exceed chunk_size.

    Some tokenizers (e.g. tiktoken) encode a leading space together with the
    following word as a single token (" foo" -> 1 token) while the bare word
    tokenizes differently ("foo" -> 2 tokens). Since TokenTextSplitter strips
    leading/trailing whitespace from each emitted chunk (unless
    keep_whitespaces=True), the accumulated token count used to decide when
    to close a chunk could previously undercount by the amount the strip
    inflates the first word's token count, letting the actual emitted chunk
    silently exceed chunk_size. Parametrized over a few independently
    generated (text, chunk_size) pairs so this checks the general invariant
    rather than one specific boundary.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base").encode
    splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=tokenizer
    )

    chunks = splitter.split_text(text)

    for chunk in chunks:
        assert len(tokenizer(chunk)) <= chunk_size, (
            f"chunk exceeds chunk_size={chunk_size}: "
            f"{len(tokenizer(chunk))} tokens -> {chunk!r}"
        )
