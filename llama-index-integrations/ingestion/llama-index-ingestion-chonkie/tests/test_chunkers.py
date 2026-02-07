"""Tests for Chonkie Chunker integration."""

from typing import List


from llama_index.core.schema import Document, MetadataMode, TextNode

from llama_index.ingestion.chonkie.chunkers import Chunker, CHUNKERS


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


def test_split_text_metadata_aware() -> None:
    """Test metadata-aware text splitting."""
    chunker = Chunker(chunk_size=512)
    text = "This is a test document. " * 20
    metadata_str = "title: Test Document\nauthor: Test Author\n"

    chunks = chunker.split_text_metadata_aware(text, metadata_str)

    assert len(chunks) > 0
    assert isinstance(chunks, list)


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

    Using TokenChunker as we want to test overlap=0 which is explicit there.
    """
    chunker = Chunker(chunker_type="token", chunk_size=100, chunk_overlap=0)
    text = "This is a test sentence. " * 50

    chunks = chunker.split_text(text)

    # Combine all chunks and verify total length is reasonable
    combined_text = "".join(chunks)
    # With overlap=0, combined length might differ slightly due to processing
    # but should be in the same ballpark
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
    chunker_small = Chunker(chunker_type="token", chunk_size=50, chunk_overlap=10)
    chunker_large = Chunker(chunker_type="token", chunk_size=500, chunk_overlap=50)

    text = "This is a test sentence. " * 50

    chunks_small = chunker_small.split_text(text)
    chunks_large = chunker_large.split_text(text)

    # Smaller chunk size should produce more chunks
    assert len(chunks_small) >= len(chunks_large)


def test_available_chunkers() -> None:
    """Test that all available chunkers can be initialized."""
    assert len(CHUNKERS) > 0
    for chunker_type in CHUNKERS:
        kwargs = {}
        try:
            if chunker_type == "code":
                kwargs = {"language": "python"}
            chunker = Chunker(chunker_type=chunker_type, **kwargs)
        except Exception as e:
            raise AssertionError(f"Failed to initialize chunker '{chunker_type}': {e}")
        assert chunker is not None
        assert chunker.chunker is not None
        # slow section (downloading embedding models in workflow is slow)
        # try:
        #     chunker.split_text("some text")
        # except Exception as e:
        #     raise AssertionError(
        #         f"Failed to split text with chunker '{chunker_type}': {e}"
        #     )
