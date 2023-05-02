"""Test pinecone indexes."""

import sys
from typing import List
from unittest.mock import MagicMock, Mock

import pytest
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import GPTVectorStoreIndex

from llama_index.readers.schema.base import Document
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from tests.indices.vector_store.utils import MockPineconeIndex

from tests.mock_utils.mock_utils import mock_tokenizer


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
    return [Document(doc_text)]


def test_build_pinecone(
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test build GPTVectorStoreIndex with PineconeVectorStore."""
    # NOTE: mock pinecone import
    sys.modules["pinecone"] = MagicMock()
    # NOTE: mock pinecone index
    pinecone_index = MockPineconeIndex()

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, tokenizer=Mock())
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = GPTVectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=mock_service_context,
        tokenizer=mock_tokenizer,
    )

    retriever = index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve("What is?")
    assert len(nodes) == 1
    assert nodes[0].node.get_text() == "This is another test."
