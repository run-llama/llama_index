

from llama_index.text_splitter import SentenceSplitter, TokenTextSplitter


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
