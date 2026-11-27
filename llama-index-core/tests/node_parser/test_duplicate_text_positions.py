"""Test that node parsers assign unique positions to duplicate text."""

import pytest

from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.schema import Document, TextNode, MetadataMode


def _validate_nodes(nodes: list[TextNode], doc: Document) -> None:
    """Validate node positions: no duplicates, valid bounds, text matches."""
    position_map = {}
    prev_start = -1

    for i, node in enumerate(nodes):
        if not isinstance(node, TextNode):
            continue

        start, end = node.start_char_idx, node.end_char_idx
        text = node.get_content(metadata_mode=MetadataMode.NONE)

        # Valid bounds and no negative lengths
        assert start is not None and end is not None
        assert 0 <= start <= end <= len(doc.text)

        # Text matches document
        assert doc.text[start:end] == text, f"Node {i} text mismatch at [{start}:{end}]"

        # No duplicate positions
        key = (start, end, text)
        assert key not in position_map, (
            f"Nodes {position_map[key]} and {i} have duplicate position [{start}:{end}]"
        )
        position_map[key] = i

        # Sequential ordering
        assert start >= prev_start, f"Node {i} out of order"
        prev_start = start


def test_markdown_with_duplicate_headers():
    """Test MarkdownNodeParser with repeated section headers."""
    doc = Document(
        text="""# Title

## Introduction
Content A.

## Methods
Content B.

## Introduction
Content C.

## Introduction
Content D.
""",
        doc_id="test",
    )

    nodes = MarkdownNodeParser().get_nodes_from_documents([doc])
    _validate_nodes(nodes, doc)


def test_sentence_splitter_with_duplicates():
    """Test SentenceSplitter with repeated sentences."""
    doc = Document(
        text=(
            "This is important. Other text. This is important. "
            "More text. This is important. Final text."
        ),
        doc_id="test",
    )

    parser = SentenceSplitter(chunk_size=50, chunk_overlap=0, include_metadata=False)
    nodes = parser.get_nodes_from_documents([doc])
    _validate_nodes(nodes, doc)


def test_multiple_documents():
    """Test position tracking is independent per document."""
    text = "## Section A\nContent.\n\n## Section A\nMore content."
    docs = [Document(text=text, doc_id=f"doc{i}") for i in range(2)]

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(docs)

    for doc in docs:
        doc_nodes = [
            n for n in nodes if isinstance(n, TextNode) and n.ref_doc_id == doc.doc_id
        ]
        _validate_nodes(doc_nodes, doc)


@pytest.mark.parametrize(
    ("splitter_cls", "chunk_size"), [(SentenceSplitter, 2), (TokenTextSplitter, 4)]
)
@pytest.mark.parametrize(
    ("overlap", "expected"),
    [
        (0, [("red red", 0, 7), ("red", 8, 11)]),
        (1, [("red red", 0, 7), ("red red", 4, 11)]),
    ],
)
def test_repeated_words_respect_chunk_overlap(
    splitter_cls: type[SentenceSplitter] | type[TokenTextSplitter],
    chunk_size: int,
    overlap: int,
    expected: list[tuple[str, int, int]],
) -> None:
    splitter = splitter_cls(chunk_size=chunk_size, chunk_overlap=overlap)
    nodes = splitter.get_nodes_from_documents([Document(text="red red red")])

    assert [
        (n.get_content(), n.start_char_idx, n.end_char_idx) for n in nodes
    ] == expected


def test_zero_token_punctuation_overlap_preserves_offsets() -> None:
    splitter = TokenTextSplitter(
        tokenizer=lambda s: list(s.replace(".", "").replace(" ", "")),
        separator=".",
        chunk_size=5,
        chunk_overlap=0,
        keep_whitespaces=True,
    )
    nodes = splitter.get_nodes_from_documents([Document(text="bb .. cc")])

    assert [(n.get_content(), n.start_char_idx, n.end_char_idx) for n in nodes] == [
        ("bb .", 0, 4),
        (".. cc", 3, 8),
    ]
