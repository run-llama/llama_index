import tiktoken

from llama_index.text_splitter import SentenceSplitter


def test_paragraphs() -> None:
    """Test case of a string with multiple paragraphs."""
    sentence_text_splitter = SentenceSplitter(chunk_size=20, chunk_overlap=0)

    text = " ".join(["foo"] * 15) + "\n\n\n" + " ".join(["bar"] * 15)
    sentence_split = sentence_text_splitter.split_text(text)
    assert sentence_split[0] == " ".join(["foo"] * 15)
    assert sentence_split[1] == " ".join(["bar"] * 15)


def test_sentences() -> None:
    """Test case of a string with multiple sentences."""
    sentence_text_splitter = SentenceSplitter(chunk_size=20, chunk_overlap=0)

    text = " ".join(["foo"] * 15) + ". " + " ".join(["bar"] * 15)
    sentence_split = sentence_text_splitter.split_text(text)

    assert sentence_split[0] == " ".join(["foo"] * 15) + "."
    assert sentence_split[1] == " ".join(["bar"] * 15)


def test_chinese_text(chinese_text: str) -> None:
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=0)
    chunks = splitter.split_text(chinese_text)
    assert len(chunks) == 3


def test_contiguous_text(contiguous_text: str) -> None:
    # NOTE: sentence splitter does not split contiguous text
    splitter = SentenceSplitter(chunk_size=100, chunk_overlap=0)
    chunks = splitter.split_text(contiguous_text)
    assert len(chunks) == 1


def test_split_with_metadata(english_text: str) -> None:
    chunk_size = 100
    metadata_str = "word " * 50
    tokenizer = tiktoken.get_encoding("gpt2")
    splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=0, tokenizer=tokenizer.encode
    )

    chunks = splitter.split_text(english_text)
    assert len(chunks) == 2

    chunks = splitter.split_text_metadata_aware(english_text, metadata_str=metadata_str)
    assert len(chunks) == 4
    for chunk in chunks:
        node_content = chunk + metadata_str
        assert len(tokenizer.encode(node_content)) <= 100
