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
    splitter = SentenceSplitter(chunk_size=100, chunk_overlap=0)
    chunks = splitter.split_text(contiguous_text)
    assert len(chunks) == 11


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


def test_edge_case() -> None:
    """Test case from: https://github.com/jerryjliu/llama_index/issues/7287"""
    text = "\n\nMarch 2020\n\nL&D Metric (Org) - 2.92%\n\n| Training Name                                                                                                          | Catergory       | Duration (hrs) | Invitees | Attendance | Target Training Hours | Actual Training Hours | Adoption % |\n| ---------------------------------------------------------------------------------------------------------------------- | --------------- | -------------- | -------- | ---------- | --------------------- | --------------------- | ---------- |\n| Overview of Data Analytics                                      | Technical       | 1              | 23       | 10         | 23                    | 10                    | 43.5       |\n| Sales & Learning Best Practices - Introduction to OTT Platforms | Technical       | 0.5            | 16       | 12         | 8                     | 6                     | 75         |\n| Leading Through OKRs                                                                                                   | Lifeskill       | 1              | 1        | 1          | 1                     | 1                     | 100        |\n| COVID: Lockdown Awareness Session                                                                                      | Lifeskill       | 2              | 1        | 1          | 2                     | 2                     | 100        |\n| Navgati Interview                                                                                                      | Lifeskill       | 2              | 6        | 6          | 12                    | 12                    | 100        |\n| leadership Summit                                               | Leadership      | 18             | 42       | 42         | 756                   | 756                   | 100        |\n| AWS - AI/ML - Online Conference                                                                                        | Project Related | 15             | 2        | 2          | 30                    | 30                    | 100        |\n"  # noqa
    splitter = SentenceSplitter()
    chunks = splitter.split_text(text)
    assert len(chunks) == 2
