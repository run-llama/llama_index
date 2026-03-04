from typing import List

import tiktoken
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.core.schema import Document, MetadataMode, TextNode


def test_paragraphs() -> None:
    """Test case of a string with multiple paragraphs."""
    sentence_text_splitter = SentenceSplitter(chunk_size=20, chunk_overlap=0)

    text = " ".join(["foo"] * 15) + "\n\n\n" + " ".join(["bar"] * 15)
    sentence_split = sentence_text_splitter.split_text(text)
    assert sentence_split[0] == " ".join(["foo"] * 15)
    assert sentence_split[1] == " ".join(["bar"] * 15)


def test_start_end_char_idx() -> None:
    document = Document(text=" ".join(["foo"] * 15) + "\n\n\n" + " ".join(["bar"] * 15))
    text_splitter = SentenceSplitter(chunk_size=2, chunk_overlap=1)
    nodes: List[TextNode] = text_splitter.get_nodes_from_documents([document])
    for node in nodes:
        assert node.start_char_idx is not None
        assert node.end_char_idx is not None
        assert node.end_char_idx - node.start_char_idx == len(
            node.get_content(metadata_mode=MetadataMode.NONE)
        )


def test_start_end_char_idx_repeating_regex_chars() -> None:
    """Test case of a string with repeating characters in [,.;。？！]."""
    document = Document(text="[this is a link](../path/to/file.md) " * 12)
    text_splitter = SentenceSplitter(chunk_size=16, chunk_overlap=1)
    nodes: List[TextNode] = text_splitter.get_nodes_from_documents([document])
    for node in nodes:
        assert node.start_char_idx is not None
        assert node.end_char_idx is not None
        assert node.end_char_idx - node.start_char_idx == len(
            node.get_content(metadata_mode=MetadataMode.NONE)
        )


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
    assert len(chunks) == 2


def test_contiguous_text(contiguous_text: str) -> None:
    splitter = SentenceSplitter(chunk_size=100, chunk_overlap=0)
    chunks = splitter.split_text(contiguous_text)
    assert len(chunks) == 10
    # technically this is incorrect. The resulting chunks only
    # have 100 characters and 40 tokens, but that's a result of
    # us using the fallback character by character splitter
    # Shouldn't be a issue in normal use though.


def test_split_with_metadata(english_text: str) -> None:
    chunk_size = 100
    metadata_str = "word " * 50
    tokenizer = tiktoken.get_encoding("cl100k_base")
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
    """Test case from: https://github.com/jerryjliu/llama_index/issues/7287."""
    text = "\n\nMarch 2020\n\nL&D Metric (Org) - 2.92%\n\n| Training Name                                                                                                          | Category       | Duration (hrs) | Invitees | Attendance | Target Training Hours | Actual Training Hours | Adoption % |\n| ---------------------------------------------------------------------------------------------------------------------- | --------------- | -------------- | -------- | ---------- | --------------------- | --------------------- | ---------- |\n| Overview of Data Analytics                                      | Technical       | 1              | 23       | 10         | 23                    | 10                    | 43.5       |\n| Sales & Learning Best Practices - Introduction to OTT Platforms | Technical       | 0.5            | 16       | 12         | 8                     | 6                     | 75         |\n| Leading Through OKRs                                                                                                   | Lifeskill       | 1              | 1        | 1          | 1                     | 1                     | 100        |\n| COVID: Lockdown Awareness Session                                                                                      | Lifeskill       | 2              | 1        | 1          | 2                     | 2                     | 100        |\n| Navgati Interview                                                                                                      | Lifeskill       | 2              | 6        | 6          | 12                    | 12                    | 100        |\n| leadership Summit                                               | Leadership      | 18             | 42       | 42         | 756                   | 756                   | 100        |\n| AWS - AI/ML - Online Conference                                                                                        | Project Related | 15             | 2        | 2          | 30                    | 30                    | 100        |\n"
    splitter = SentenceSplitter(tokenizer=tiktoken.get_encoding("gpt2").encode)
    chunks = splitter.split_text(text)
    assert len(chunks) == 2

    splitter = SentenceSplitter(tokenizer=tiktoken.get_encoding("cl100k_base").encode)
    chunks = splitter.split_text(text)
    # Like the Chinese there's a big difference in the # of tokens
    assert len(chunks) == 1


def test_overlap() -> None:
    splitter = SentenceSplitter(chunk_size=15, chunk_overlap=10)
    chunks = splitter.split_text("Hello! How are you? I am fine. And you?")
    assert len(chunks) == 1

    chunks2 = splitter.split_text(
        "Hello! How are you? I am fine. And you? This is a slightly longer sentence."
    )
    assert len(chunks2) == 3
    # With the overflow fix, overlap may be reduced to ensure chunk_size is not exceeded
    assert chunks2[2] == "And you? This is a slightly longer sentence."


def test_split_texts_singleton() -> None:
    """Test case for a singleton list of texts."""
    sentence_text_splitter = SentenceSplitter(chunk_size=20, chunk_overlap=0)

    text = " ".join(["foo"] * 15) + "\n\n\n" + " ".join(["bar"] * 15)
    texts = [text]
    sentence_split = sentence_text_splitter.split_texts(texts)
    assert sentence_split[0] == " ".join(["foo"] * 15)
    assert sentence_split[1] == " ".join(["bar"] * 15)


def test_split_texts_multiple() -> None:
    """Test case for a list of texts."""
    sentence_text_splitter = SentenceSplitter(chunk_size=20, chunk_overlap=0)

    text1 = " ".join(["foo"] * 15) + "\n\n\n" + " ".join(["bar"] * 15)
    text2 = " ".join(["bar"] * 15) + "\n\n\n" + " ".join(["foo"] * 15)
    texts = [text1, text2]
    sentence_split = sentence_text_splitter.split_texts(texts)
    print(sentence_split)
    assert sentence_split[0] == " ".join(["foo"] * 15)
    assert sentence_split[1] == " ".join(["bar"] * 15)
    assert sentence_split[2] == " ".join(["bar"] * 15)
    assert sentence_split[3] == " ".join(["foo"] * 15)


def test_split_texts_with_metadata(english_text: str) -> None:
    """Test case for a list of texts with metadata."""
    chunk_size = 100
    metadata_str = "word " * 50
    tokenizer = tiktoken.get_encoding("cl100k_base")
    splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=0, tokenizer=tokenizer.encode
    )

    chunks = splitter.split_texts([english_text, english_text])
    assert len(chunks) == 4

    chunks = splitter.split_texts_metadata_aware(
        [english_text, english_text], [metadata_str, metadata_str]
    )
    assert len(chunks) == 8


def test_no_overflow_with_chinese_text_and_metadata() -> None:
    """
    Test that chunks don't exceed chunk_size even with overlap and metadata.

    This test case is from a user who reported getting 537 tokens in a chunk
    when chunk_size=512 with Chinese text and metadata.
    """
    text = """你所描述的情况可能与身体健康有关，尤其是与压力、疲劳和动机相关的身体状态。长时间的工作压力和疲劳可能导致身体功能下降，包括记忆力、注意力和决策能力。此外，焦虑和压力可能会影响你的情绪状态和工作表现，从而形成一个恶性循环。
以下是一些可能与你的情况相关的健康概念：
1\\. **慢性疲劳**：长时间的工作和缺乏休息可能导致身体的疲劳，这种慢性疲劳可能会影响你的肌肉恢复和整体健康。
2\\. **营养不足**：你提到的对工作的忽视可能导致饮食不规律和营养不足，这可能会影响你的体力和精力。
3\\. **体能和耐力**：如果你的工作不再给你提供足够的体能锻炼，或者你感觉自己的体能有所下降，这可能会影响你的工作表现。
4\\. **自我照顾**：如果你忽视了对身体的照顾，比如不按时吃饭、不运动，可能会导致身体机能的下降。
5\\. **应对策略**：你可能会采取一些应对策略来处理工作压力，比如依赖咖啡或能量饮料来提神，或者熬夜来完成工作。
为了应对这些挑战，你可以尝试以下策略：
\\- **休息和恢复**：确保你有足够的休息时间，这对于恢复体力和精神状态至关重要。
\\- **时间管理和优先级设定**：尝试合理规划你的时间，优先处理最重要的任务。
\\- **寻求支持**：和家人、朋友或同事交流你的感受，或者寻求专业的健康咨询。
\\- **自我反思**：思考你的生活方式和工作习惯，以及它们是否对你的健康有益。
\\- **健康规划**：考虑你的长期健康规划，是否需要调整你的生活方式或寻求更健康的习惯。
\\- **身体保健**：如果可能，尝试一些提高身体机能的活动，如瑜伽、太极或其他健身课程。
记住，你的身体健康是生活的基础。如果工作压力和疲劳影响了你的生活质量，那么采取行动来改变这种状况是至关重要的。专业的健康支持可能会对你有所帮助。"""

    doc = Document(
        text=text,
        metadata={
            "title": "教育的主要性 教育是人类社会发展的基石",
            "keywords": "教育、 文化、 学习、 人才、 成长、 创造、 未来、 资源、 关注、 才华和潜力",
        },
    )

    parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = parser.get_nodes_from_documents([doc])

    # Before the fix, this would produce [441, 537] - second chunk exceeds 512!
    # After the fix, all chunks should be <= 512 tokens
    for i, node in enumerate(nodes):
        content_length = len(parser._tokenizer(node.get_content(MetadataMode.ALL)))
        assert content_length <= 512, (
            f"Node {i} has {content_length} tokens, exceeds chunk_size of 512. "
            f"This indicates the overflow bug is not fixed."
        )
