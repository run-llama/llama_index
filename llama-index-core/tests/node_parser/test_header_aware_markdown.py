"""Tests for HeaderAwareMarkdownSplitter."""

from llama_index.core.node_parser.file.header_aware_markdown import (
    HeaderAwareMarkdownSplitter,
)
from llama_index.core.schema import Document


def _split(text: str, **kwargs) -> list:
    """Helper — split a markdown string and return the resulting nodes."""
    parser = HeaderAwareMarkdownSplitter(**kwargs)
    return parser.get_nodes_from_documents([Document(text=text)])


# ------------------------------------------------------------------
# Basic behaviour
# ------------------------------------------------------------------


def test_single_section_within_limit():
    """A short section stays as one node."""
    nodes = _split("# Hello\n\nWorld", chunk_size=1024)
    assert len(nodes) == 1
    assert nodes[0].text == "# Hello\n\nWorld"
    assert nodes[0].metadata["header_path"] == "/"


def test_multiple_sections_within_limit():
    """Multiple headers produce one node each when they fit."""
    md = "# A\n\nContent A\n\n# B\n\nContent B"
    nodes = _split(md, chunk_size=1024)
    assert len(nodes) == 2
    assert "Content A" in nodes[0].text
    assert "Content B" in nodes[1].text


def test_nested_headers_path():
    """Header path metadata reflects the ancestor hierarchy."""
    md = "# Top\n\nIntro\n\n## Sub\n\nSub content\n\n### Deep\n\nDeep content"
    nodes = _split(md, chunk_size=1024)
    assert len(nodes) == 3
    assert nodes[0].metadata["header_path"] == "/"
    assert nodes[1].metadata["header_path"] == "/Top/"
    assert nodes[2].metadata["header_path"] == "/Top/Sub/"


# ------------------------------------------------------------------
# Oversized section splitting
# ------------------------------------------------------------------


def test_oversized_section_splits():
    """A section exceeding chunk_size is split into multiple nodes."""
    # Build a section with many paragraphs that exceeds a tiny chunk_size.
    paragraphs = [f"Paragraph {i}. " * 10 for i in range(10)]
    body = "\n\n".join(paragraphs)
    md = f"# Big Section\n\n{body}"

    nodes = _split(md, chunk_size=50)
    assert len(nodes) > 1
    # Every chunk should start with the header when include_header_in_chunks=True.
    for node in nodes:
        assert node.text.startswith("# Big Section")


def test_oversized_section_no_header_prepend():
    """When include_header_in_chunks=False, sub-chunks don't get the header."""
    paragraphs = [f"Paragraph {i}. " * 10 for i in range(10)]
    body = "\n\n".join(paragraphs)
    md = f"# Big Section\n\n{body}"

    nodes = _split(md, chunk_size=50, include_header_in_chunks=False)
    assert len(nodes) > 1
    # First node has the header (it's part of the original section),
    # but subsequent nodes should NOT start with "#".
    non_first = [n for n in nodes if not n.text.startswith("# Big Section")]
    assert len(non_first) > 0


def test_oversized_single_paragraph_splits_by_sentences():
    """A single giant paragraph is split at sentence boundaries."""
    sentences = [f"Sentence number {i}." for i in range(50)]
    md = "# Title\n\n" + " ".join(sentences)

    nodes = _split(md, chunk_size=30)
    assert len(nodes) > 1


# ------------------------------------------------------------------
# Code blocks
# ------------------------------------------------------------------


def test_headers_in_code_blocks_ignored():
    """Headers inside backtick code blocks should not be treated as section boundaries."""
    md = "# Real Header\n\nSome text\n\n```python\n# This is a comment\ndef foo():\n    pass\n```\n\nMore text"
    nodes = _split(md, chunk_size=1024)
    assert len(nodes) == 1
    assert "# This is a comment" in nodes[0].text


def test_headers_in_tilde_code_blocks_ignored():
    """Headers inside tilde-fenced code blocks should not be treated as section boundaries."""
    md = "# Real Header\n\nSome text\n\n~~~python\n# This is a comment\ndef foo():\n    pass\n~~~\n\nMore text"
    nodes = _split(md, chunk_size=1024)
    assert len(nodes) == 1
    assert "# This is a comment" in nodes[0].text


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_empty_document():
    """Empty input produces no nodes."""
    nodes = _split("", chunk_size=1024)
    assert len(nodes) == 0


def test_no_headers():
    """Text without any headers is returned as a single node."""
    md = "Just some plain text\n\nWith paragraphs"
    nodes = _split(md, chunk_size=1024)
    assert len(nodes) == 1
    assert "Just some plain text" in nodes[0].text


def test_header_with_no_body():
    """A header followed immediately by another header produces a node with just the header."""
    md = "# First\n\n# Second\n\nContent"
    nodes = _split(md, chunk_size=1024)
    assert len(nodes) == 2
    assert nodes[0].text == "# First"
    assert "Content" in nodes[1].text


def test_oversized_single_sentence_word_fallback():
    """A single sentence exceeding chunk_size is split at word boundaries."""
    # One very long sentence with no periods in the middle.
    long_sentence = " ".join([f"word{i}" for i in range(200)])
    md = f"# Title\n\n{long_sentence}"

    nodes = _split(md, chunk_size=20)
    assert len(nodes) > 1
    # Every chunk should be non-empty.
    for node in nodes:
        assert node.text.strip()


def test_determinism():
    """Same input always produces the same splits."""
    md = "# A\n\nText A\n\n## B\n\nText B\n\n# C\n\nText C"
    nodes_1 = _split(md, chunk_size=1024)
    nodes_2 = _split(md, chunk_size=1024)
    assert [n.text for n in nodes_1] == [n.text for n in nodes_2]


# ------------------------------------------------------------------
# Custom separator
# ------------------------------------------------------------------


def test_custom_header_path_separator():
    """Custom separator is reflected in header_path metadata."""
    md = "# Top\n\nIntro\n\n## Sub\n\nContent"
    nodes = _split(md, chunk_size=1024, header_path_separator="›")
    assert nodes[1].metadata["header_path"] == "›Top›"


# ------------------------------------------------------------------
# Sub-splitter extensibility
# ------------------------------------------------------------------


def test_custom_sub_splitter():
    """A custom sub_splitter callable is used for oversized sections."""
    calls = []

    def my_splitter(text: str, max_tokens: int) -> list[str]:
        calls.append((text, max_tokens))
        return [text[:50], text[50:]]

    paragraphs = ["Word " * 100 for _ in range(5)]
    md = "# Big\n\n" + "\n\n".join(paragraphs)

    nodes = _split(md, chunk_size=20, sub_splitter=my_splitter)
    assert len(calls) > 0  # Our custom splitter was invoked
    assert len(nodes) >= 2
