"""Test pinecone indexes."""

from typing import List

import pytest

from llama_index.data_structs.node import Node
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.readers.schema.base import Document
from tests.indices.vector_store.utils import get_pinecone_storage_context
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
    """Test build VectorStoreIndex with PineconeVectorStore."""
    storage_context = get_pinecone_storage_context()
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=mock_service_context,
        tokenizer=mock_tokenizer,
    )

    retriever = index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve("What is?")
    assert len(nodes) == 1
    assert nodes[0].node.get_text() == "This is another test."


def test_node_with_metadata(
    mock_service_context: ServiceContext,
) -> None:
    storage_context = get_pinecone_storage_context()
    input_nodes = [Node(text="test node text", extra_info={"key": "value"})]
    index = VectorStoreIndex(
        input_nodes,
        storage_context=storage_context,
        service_context=mock_service_context,
    )

    retriever = index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve("What is?")
    assert len(nodes) == 1
    assert nodes[0].node.text == "test node text"
    assert nodes[0].node.extra_info == {"key": "value"}
