"""Test text splitter."""
from llama_index.langchain_helpers.text_splitter import (
    SentenceSplitter,
    TokenTextSplitter,
)
import pytest


def test_split_token() -> None:
    """Test split normal token."""
    # tiktoken will say length is ~5k
    token = "foo bar"
    text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
    chunks = text_splitter.split_text(token)
    assert chunks == ["foo", "bar"]

    token = "foo bar hello world"
    text_splitter = TokenTextSplitter(chunk_size=2, chunk_overlap=1)
    chunks = text_splitter.split_text(token)
    assert chunks == ["foo bar", "bar hello", "hello world"]


def test_truncate_token() -> None:
    """Test truncate normal token."""
    # tiktoken will say length is ~5k
    token = "foo bar"
    text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
    chunks = text_splitter.truncate_text(token)
    assert chunks == "foo"


def test_split_long_token() -> None:
    """Test split a really long token."""
    # tiktoken will say length is ~5k
    token = "a" * 100
    text_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    chunks = text_splitter.split_text(token)
    # each text chunk may have spaces, since we join splits by separator
    assert "".join(chunks).replace(" ", "") == token

    token = ("a" * 49) + "\n" + ("a" * 50)
    text_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    chunks = text_splitter.split_text(token)
    assert len(chunks[0]) == 49
    assert len(chunks[1]) == 50


def test_split_with_metadata_str() -> None:
    """Test split while taking into account chunk size used by metadata str."""
    text = " ".join(["foo"] * 20)
    metadata_str = "test_metadata_str"

    text_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    assert len(chunks) == 1

    text_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    chunks = text_splitter.split_text(text, metadata_str=metadata_str)
    assert len(chunks) == 2


def test_split_diff_sentence_token() -> None:
    """Test case of a string that will split differently."""
    token_text_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    sentence_text_splitter = SentenceSplitter(chunk_size=20, chunk_overlap=0)

    text = " ".join(["foo"] * 15) + "\n\n\n" + " ".join(["bar"] * 15)
    token_split = token_text_splitter.split_text(text)
    sentence_split = sentence_text_splitter.split_text(text)
    assert token_split[0] == " ".join(["foo"] * 15) + "\n\n\n" + " ".join(["bar"] * 3)
    assert token_split[1] == " ".join(["bar"] * 12)
    assert sentence_split[0] == " ".join(["foo"] * 15)
    assert sentence_split[1] == " ".join(["bar"] * 15)


def test_split_diff_sentence_token2() -> None:
    """Test case of a string that will split differently."""
    token_text_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    sentence_text_splitter = SentenceSplitter(chunk_size=20, chunk_overlap=0)

    text = " ".join(["foo"] * 15) + ". " + " ".join(["bar"] * 15)
    token_split = token_text_splitter.split_text(text)
    sentence_split = sentence_text_splitter.split_text(text)

    assert token_split[0] == " ".join(["foo"] * 15) + ". " + " ".join(["bar"] * 4)
    assert token_split[1] == " ".join(["bar"] * 11)
    assert sentence_split[0] == " ".join(["foo"] * 15) + "."
    assert sentence_split[1] == " ".join(["bar"] * 15)


def test_sentence_splitter_infinite_loop() -> None:
    sentence_splitter = SentenceSplitter(
        chunk_size=20,
        chunk_overlap=5,
        secondary_chunking_regex="[^\,]+[,]",
    )
    # This is ok
    sentence_splitter.split_text_with_overlaps("This is fine. No problems.")

    # This next one infinite loops.
    # The entire text is too large to be saved as a single sentence, so we fall
    # back to `secondary_chunking_regex` This splits into two chunks based on
    # the comma.
    #
    # The first split is 18 tokens, which is smaller than the chunk size (20)
    # but larger than the chunk size - chunk overlap (15)
    with pytest.raises(ValueError, match="Single token exceed chunk size"):
        sentence_splitter.split_text_with_overlaps(
            "This is a sentence hand-crafted to be very very bad for the sentence splitter, since the secondary_chunking_regex won't get the resulting splits small enough"  # noqa: E501
        )
