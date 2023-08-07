"""Test text splitter."""
from llama_index.text_splitter import TokenTextSplitter
from llama_index.text_splitter.utils import truncate_text


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
    chunks = truncate_text(token, text_splitter)
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
