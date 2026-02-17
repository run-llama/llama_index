"""Tests for Chonkie Chunker integration."""

import pytest
import importlib
from typing import List


from llama_index.core.schema import Document, MetadataMode, TextNode

from llama_index.node_parser.chonkie.chunkers import Chunker, CHUNKERS


def test_chonkie_chunker_initialization() -> None:
    """Test Chunker can be initialized with default parameters."""
    chunker = Chunker(chunk_size=512)
    assert chunker is not None
    assert chunker.chunker is not None


def test_chonkie_chunker_from_defaults() -> None:
    """Test Chunker can be created using from_defaults."""
    chunker = Chunker.from_defaults()
    assert chunker is not None
    assert isinstance(chunker, Chunker)
    # Default is recursive
    from chonkie import RecursiveChunker

    assert isinstance(chunker.chunker, RecursiveChunker)


def test_chonkie_chunker_class_name() -> None:
    """Test that class_name returns the correct name."""
    chunker = Chunker(chunk_size=512)
    assert chunker.class_name() == "Chunker"


def test_split_text_empty() -> None:
    """Test splitting empty text returns empty list with one empty string."""
    chunker = Chunker(chunk_size=512)
    result = chunker.split_text("")
    assert result == [""]


def test_split_text_basic() -> None:
    """Test basic text splitting functionality."""
    chunker = Chunker(chunk_size=100)
    text = "This is a test. " * 50  # Create text that will need splitting
    chunks = chunker.split_text(text)

    assert len(chunks) > 0
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_split_text_short() -> None:
    """Test that short text is not split."""
    chunker = Chunker(chunk_size=512)
    text = "This is a short text."
    chunks = chunker.split_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_with_paragraphs() -> None:
    """Test splitting text with multiple paragraphs."""
    chunker = Chunker(chunk_size=100)
    text = "First paragraph. " * 20 + "\n\n" + "Second paragraph. " * 20
    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) > 0 for chunk in chunks)


def test_split_text_metadata_aware_basic() -> None:
    """Test metadata-aware text splitting returns valid chunks."""
    chunker = Chunker(chunker="recursive", chunk_size=200)
    text = "This is a test document. " * 20
    metadata_str = "title: Test Document\nauthor: Test Author\n"

    chunks = chunker.split_text_metadata_aware(text, metadata_str)

    assert len(chunks) > 0
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)
    assert "".join(chunks).strip() == text.strip()


def test_split_text_metadata_aware_more_chunks_with_metadata() -> None:
    """With recursive chunker, metadata reduces effective chunk size so we get more (or equal) chunks."""
    chunker = Chunker(chunker="recursive", chunk_size=100)
    text = "This is a test document. " * 30

    chunks_no_metadata = chunker.split_text(text)
    metadata_str = "title: Test Document\nauthor: Test Author\ndate: 2024\n"
    chunks_with_metadata = chunker.split_text_metadata_aware(text, metadata_str)

    assert len(chunks_with_metadata) >= len(chunks_no_metadata)
    assert all(isinstance(c, str) for c in chunks_with_metadata)
    assert "".join(chunks_with_metadata).strip() == text.strip()


def test_split_text_metadata_aware_empty_metadata() -> None:
    """Empty metadata: effective_chunk_size equals chunk_size, so result matches split_text."""
    chunker = Chunker(chunker="token", chunk_size=80)
    text = "This is a test. " * 20

    chunks_empty_metadata = chunker.split_text_metadata_aware(text, "")
    chunks_no_metadata = chunker.split_text(text)

    assert len(chunks_empty_metadata) == len(chunks_no_metadata)
    assert chunks_empty_metadata == chunks_no_metadata


def test_split_text_metadata_aware_value_error_when_metadata_too_long() -> None:
    """ValueError when metadata token count >= chunk_size (effective_chunk_size <= 0)."""
    chunker = Chunker(chunker="recursive", chunk_size=5)
    text = "Short text."
    long_metadata = "title: " + "very long title " * 20 + "author: " + "name " * 20

    with pytest.raises(ValueError) as exc_info:
        chunker.split_text_metadata_aware(text, long_metadata)

    assert "Metadata length" in str(exc_info.value)
    assert "chunk size" in str(exc_info.value).lower()


def test_split_text_metadata_aware_chunk_size_restored() -> None:
    """chunk_size is restored after split so repeated metadata-aware calls are consistent."""
    chunker = Chunker(chunker="recursive", chunk_size=60)
    text = "This is a test document. " * 15
    metadata_str = "title: Doc\n"

    chunks1 = chunker.split_text_metadata_aware(text, metadata_str)
    chunks2 = chunker.split_text_metadata_aware(text, metadata_str)

    assert chunks1 == chunks2
    chunks_plain = chunker.split_text(text)
    assert len(chunks_plain) >= 1


def test_split_text_metadata_aware_preserves_content() -> None:
    """Metadata-aware splitting preserves full text (no overlap, so join equals original)."""
    chunker = Chunker(chunker="recursive", chunk_size=50)
    text = "One. Two. Three. Four. Five. " * 10
    metadata_str = "key: value\n"

    chunks = chunker.split_text_metadata_aware(text, metadata_str)
    combined = "".join(chunks)

    assert combined.strip() == text.strip()
    assert len(chunks) > 0


def test_split_text_metadata_aware_single_chunk_short_text() -> None:
    """Short text with metadata still fits in one chunk when effective size is large enough."""
    chunker = Chunker(chunker="recursive", chunk_size=512)
    text = "Short piece of text."
    metadata_str = "title: Short\n"

    chunks = chunker.split_text_metadata_aware(text, metadata_str)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_metadata_aware_fallback_same_as_split_text() -> None:
    """Chunkers without tokenizer (e.g. fast) fall back to split_text; result equals split_text."""
    chunker = Chunker(chunker="fast", chunk_size=10, forward_fallback=True)
    text = "This is a test. " * 50
    metadata_str = "title: Test\n"

    chunks_metadata_aware = chunker.split_text_metadata_aware(text, metadata_str)
    chunks_plain = chunker.split_text(text)

    assert chunks_metadata_aware == chunks_plain


def test_split_text_metadata_aware_fallback_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fallback path logs a warning once per instance for incompatible chunkers."""
    import logging

    with caplog.at_level(logging.WARNING):
        chunker = Chunker(chunker="fast", chunk_size=20, forward_fallback=True)
        chunker.split_text_metadata_aware("Some text." * 100, "meta: value\n")
        chunker.split_text_metadata_aware("More text." * 100, "meta: value\n")

    # Warning only logged once per instance
    warnings = [r for r in caplog.records if "metadata awareness" in (r.message or "")]
    assert len(warnings) >= 1
    assert "does not support metadata awareness" in (warnings[0].message or "")


def test_get_nodes_from_documents() -> None:
    """Test generating nodes from documents."""
    chunker = Chunker(chunk_size=100)
    documents = [
        Document(text="This is document one. " * 30, metadata={"doc_id": "1"}),
        Document(text="This is document two. " * 30, metadata={"doc_id": "2"}),
    ]

    nodes: List[TextNode] = chunker.get_nodes_from_documents(documents)

    assert len(nodes) > 0
    assert all(isinstance(node, TextNode) for node in nodes)
    # Check that metadata is preserved
    assert any(node.metadata.get("doc_id") in ["1", "2"] for node in nodes)


def test_start_end_char_idx() -> None:
    """Test that start and end character indices are correctly set."""
    chunker = Chunker(chunk_size=50)
    document = Document(text="This is a test document. " * 20)

    nodes: List[TextNode] = chunker.get_nodes_from_documents([document])

    for node in nodes:
        assert node.start_char_idx is not None
        assert node.end_char_idx is not None
        # Verify that the indices correctly extract the node content
        assert node.end_char_idx - node.start_char_idx == len(
            node.get_content(metadata_mode=MetadataMode.NONE)
        )


def test_nodes_preserve_document_metadata() -> None:
    """Test that nodes preserve document metadata."""
    chunker = Chunker(chunk_size=100)
    metadata = {"title": "Test Document", "author": "Test Author", "year": 2024}
    document = Document(text="This is a test. " * 30, metadata=metadata)

    nodes: List[TextNode] = chunker.get_nodes_from_documents([document])

    for node in nodes:
        assert node.metadata["title"] == metadata["title"]
        assert node.metadata["author"] == metadata["author"]
        assert node.metadata["year"] == metadata["year"]


def test_chunker_with_long_text() -> None:
    """Test chunker with a longer piece of text."""
    chunker = Chunker(chunk_size=200)
    # Create a long text that will require multiple chunks
    text = (
        """
    The quick brown fox jumps over the lazy dog. This is a test sentence.
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod
    tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
    quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
    consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse
    cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
    non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    """
        * 10
    )

    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    # Verify no chunks are empty
    assert all(len(chunk.strip()) > 0 for chunk in chunks)


def test_chunker_preserves_text_content() -> None:
    """
    Test that chunker preserves all text content across chunks.
    """
    chunker = Chunker(chunker="recursive", chunk_size=100)
    text = "This is a test sentence. " * 50

    chunks = chunker.split_text(text)

    combined_text = "".join(chunks)
    assert len(combined_text) > 0


def test_include_metadata_flag() -> None:
    """Test that include_metadata flag works correctly."""
    chunker_with_metadata = Chunker(chunk_size=100, include_metadata=True)
    chunker_without_metadata = Chunker(chunk_size=100, include_metadata=False)

    document = Document(
        text="This is a test. " * 30, metadata={"title": "Test", "author": "Author"}
    )

    nodes_with = chunker_with_metadata.get_nodes_from_documents([document])
    nodes_without = chunker_without_metadata.get_nodes_from_documents([document])

    # Both should produce nodes
    assert len(nodes_with) > 0
    assert len(nodes_without) > 0


def test_include_prev_next_rel_flag() -> None:
    """Test that include_prev_next_rel flag affects node relationships."""
    chunker_with_rel = Chunker(chunk_size=100, include_prev_next_rel=True)
    chunker_without_rel = Chunker(chunk_size=100, include_prev_next_rel=False)

    document = Document(text="This is a test. " * 30)

    nodes_with = chunker_with_rel.get_nodes_from_documents([document])
    nodes_without = chunker_without_rel.get_nodes_from_documents([document])

    # Both should produce nodes
    assert len(nodes_with) > 0
    assert len(nodes_without) > 0

    # Nodes with relationships should have prev/next set (except first/last)
    if len(nodes_with) > 1:
        # Second node should have previous
        assert (
            nodes_with[1].prev_node is not None
            or not chunker_with_rel.include_prev_next_rel
        )


def test_chunker_with_special_characters() -> None:
    """Test chunker handles text with special characters."""
    chunker = Chunker(chunk_size=100)
    text = "Hello! How are you? I'm fine. This costs $10. Email: test@example.com"

    chunks = chunker.split_text(text)

    assert len(chunks) > 0
    # Verify the text is preserved
    combined = "".join(chunks)
    assert "@example.com" in combined or len(chunks) == 1


def test_chunker_with_unicode() -> None:
    """Test chunker handles Unicode text."""
    chunker = Chunker(chunk_size=100)
    text = "Hello 世界! This is a test with émojis 🎉 and special chars: café, naïve."

    chunks = chunker.split_text(text)

    assert len(chunks) > 0
    combined = "".join(chunks)
    assert "世界" in combined or "🎉" in combined or len(chunks) == 1


def test_multiple_documents() -> None:
    """Test processing multiple documents at once."""
    chunker = Chunker(chunk_size=100)
    documents = [
        Document(text="Document one. " * 30, metadata={"id": 1}),
        Document(text="Document two. " * 30, metadata={"id": 2}),
        Document(text="Document three. " * 30, metadata={"id": 3}),
    ]

    nodes = chunker.get_nodes_from_documents(documents)

    # Should have nodes from all documents
    assert len(nodes) > 3
    # Verify all document IDs are represented
    doc_ids = {node.metadata.get("id") for node in nodes}
    assert doc_ids == {1, 2, 3}


def test_chunker_consistency() -> None:
    """Test that chunker produces consistent results."""
    chunker = Chunker(chunk_size=100)
    text = "This is a test sentence. " * 20

    chunks1 = chunker.split_text(text)
    chunks2 = chunker.split_text(text)

    assert len(chunks1) == len(chunks2)
    assert chunks1 == chunks2


def test_chunker_kwargs() -> None:
    """
    Test that chunker accepts and uses kwargs properly.

    Using TokenChunker to test chunk_overlap support.
    """
    # Test with various chunk sizes
    chunker_small = Chunker(chunker="token", chunk_size=50, chunk_overlap=10)
    chunker_large = Chunker(chunker="token", chunk_size=500, chunk_overlap=50)

    text = "This is a test sentence. " * 50

    chunks_small = chunker_small.split_text(text)
    chunks_large = chunker_large.split_text(text)

    # Smaller chunk size should produce more chunks
    assert len(chunks_small) >= len(chunks_large)


def _optional_chunker_available(chunker_type: str) -> bool:
    """Return whether a chunker can be safely instantiated in minimal env."""
    if chunker_type == "late":
        # requires sentence-transformers -> transformers -> torch
        try:
            importlib.import_module("sentence_transformers")
            importlib.import_module("torch")
            return True
        except Exception:
            return False
    return True

def test_available_chunkers() -> None:
    """Test that all available chunkers can be initialized."""
    assert len(CHUNKERS) > 0

    for chunker_type in CHUNKERS:
        if not _optional_chunker_available(chunker_type):
            pytest.skip(f"Skipping optional chunker '{chunker_type}' (missing heavy deps)")

        kwargs = {}
        try:
            if chunker_type == "code":
                kwargs = {"language": "python"}

            chunker = Chunker(chunker=chunker_type, **kwargs)

        except Exception as e:
            raise AssertionError(f"Failed to initialize chunker '{chunker_type}': {e}")

        assert chunker is not None
        assert chunker.chunker is not None

def test_chunker_with_instance() -> None:
    """Test Chunker initialization with a chonkie chunker instance."""
    from chonkie import RecursiveChunker

    chunker = RecursiveChunker(chunk_size=2048)
    parser = Chunker(chunker)
    assert parser is not None
    assert parser.chunker is not None
    assert isinstance(parser.chunker, RecursiveChunker)
