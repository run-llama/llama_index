"""Test node utils."""

from typing import List

import pytest

from llama_index.node_parser.node_utils import TextSplit, get_nodes_from_document
from llama_index.schema import Document, MetadataMode
from llama_index.text_splitter import TokenTextSplitter


class TokenTextSplitterWithMetadata(TokenTextSplitter):
    """Text splitter which adds metadata to text splits."""

    def _postprocess_splits(self, docs: List[TextSplit]) -> List[TextSplit]:
        for doc in docs:
            doc.metadata = {"test_splitter_key": "test_splitter_val"}

        docs = super()._postprocess_splits(docs)

        return docs


@pytest.fixture
def text_splitter() -> TokenTextSplitter:
    """Get text splitter."""
    return TokenTextSplitter(chunk_size=20, chunk_overlap=0)


@pytest.fixture
def text_splitter_with_metadata() -> TokenTextSplitterWithMetadata:
    """Get text splitter which adds metadata."""
    return TokenTextSplitterWithMetadata(chunk_size=20, chunk_overlap=0)


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [
        Document(text=doc_text, id_="test_doc_id", metadata={"test_key": "test_val"})
    ]


def test_get_nodes_from_document(
    documents: List[Document], text_splitter: TokenTextSplitter
) -> None:
    """Test get nodes from document have desired chunk size."""
    nodes = get_nodes_from_document(
        documents[0],
        text_splitter,
        include_metadata=False,
    )
    assert len(nodes) == 2
    actual_chunk_sizes = [
        len(text_splitter.tokenizer(node.get_content())) for node in nodes
    ]
    assert all(
        chunk_size <= text_splitter._chunk_size for chunk_size in actual_chunk_sizes
    )


def test_get_nodes_from_document_with_metadata(
    documents: List[Document], text_splitter: TokenTextSplitter
) -> None:
    """Test get nodes from document with metadata have desired chunk size."""
    nodes = get_nodes_from_document(
        documents[0],
        text_splitter,
        include_metadata=True,
    )
    assert len(nodes) == 3
    actual_chunk_sizes = [
        len(text_splitter.tokenizer(node.get_content())) for node in nodes
    ]
    assert all(
        chunk_size <= text_splitter._chunk_size for chunk_size in actual_chunk_sizes
    )
    assert all(
        [
            "test_key: test_val" in n.get_content(metadata_mode=MetadataMode.ALL)
            for n in nodes
        ]
    )


def test_get_nodes_from_document_with_node_metadata(
    documents: List[Document],
    text_splitter_with_metadata: TokenTextSplitterWithMetadata,
) -> None:
    """Test get nodes from document with text splits metadata"""
    nodes = get_nodes_from_document(
        documents[0],
        text_splitter_with_metadata,
        include_metadata=True,
    )
    assert len(nodes) == 3
    assert all(
        [
            "test_key: test_val" in n.get_content(metadata_mode=MetadataMode.ALL)
            for n in nodes
        ]
    )
    assert all(
        [
            "test_splitter_key: test_splitter_val"
            in n.get_content(metadata_mode=MetadataMode.ALL)
            for n in nodes
        ]
    )
