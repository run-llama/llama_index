"""Test node utils."""

from typing import List

import pytest
import tiktoken

from llama_index.bridge.langchain import RecursiveCharacterTextSplitter
from llama_index.node_parser.node_utils import get_nodes_from_document
from llama_index.schema import Document, MetadataMode
from llama_index.text_splitter import TokenTextSplitter


@pytest.fixture
def text_splitter() -> TokenTextSplitter:
    """Get text splitter."""
    return TokenTextSplitter(chunk_size=20, chunk_overlap=0)


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
    assert len(nodes) == 1
    actual_chunk_sizes = [
        len(text_splitter.tokenizer(node.get_content())) for node in nodes
    ]
    assert all(
        chunk_size <= text_splitter.chunk_size for chunk_size in actual_chunk_sizes
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
    assert len(nodes) == 2
    actual_chunk_sizes = [
        len(text_splitter.tokenizer(node.get_content(metadata_mode=MetadataMode.ALL)))
        for node in nodes
    ]
    assert all(
        chunk_size <= text_splitter.chunk_size for chunk_size in actual_chunk_sizes
    )
    assert all(
        [
            "test_key: test_val" in n.get_content(metadata_mode=MetadataMode.ALL)
            for n in nodes
        ]
    )


def test_get_nodes_from_document_langchain_compatible(
    documents: List[Document],
) -> None:
    """Test get nodes from document have desired chunk size."""
    tokenizer = tiktoken.get_encoding("gpt2").encode
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=20, chunk_overlap=0
    )
    nodes = get_nodes_from_document(
        documents[0],
        text_splitter,  # type: ignore
        include_metadata=False,
    )
    assert len(nodes) == 2
    actual_chunk_sizes = [len(tokenizer(node.get_content())) for node in nodes]
    assert all(
        chunk_size <= text_splitter._chunk_size for chunk_size in actual_chunk_sizes
    )
