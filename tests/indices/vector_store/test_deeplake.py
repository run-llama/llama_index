"""Test deeplake indexes."""

import os
from typing import List

import pytest

from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.schema import Document
from llama_index.schema import TextNode
from llama_index.storage.storage_context import StorageContext


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    doc_text1 = "Hello world!"
    doc_text2 = "This is the first test. answer is A"
    doc_text3 = "This is the second test. answer is B"
    doc_text4 = "This is the third test. answer is C"

    return [
        Document(text=doc_text1),
        Document(text=doc_text2),
        Document(text=doc_text3),
        Document(text=doc_text4),
    ]


@pytest.mark.skipif("CI" in os.environ, reason="no DeepLake in CI")
def test_build_deeplake(
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    import deeplake

    """Test build VectorStoreIndex with DeepLakeVectorStore."""
    dataset_path = "./llama_index_test"
    vector_store = DeepLakeVectorStore(
        dataset_path=dataset_path,
        overwrite=True,
        verbose=False,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=mock_service_context,
    )

    retriever = index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve("What is the answer to the third test?")
    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "This is the third test. answer is C"
    deeplake.delete(dataset_path)


@pytest.mark.skipif("CI" in os.environ, reason="no DeepLake in CI")
def test_node_with_metadata(
    mock_service_context: ServiceContext,
) -> None:
    import deeplake

    dataset_path = "./llama_index_test"
    vector_store = DeepLakeVectorStore(
        dataset_path=dataset_path,
        overwrite=True,
        verbose=False,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    input_nodes = [TextNode(text="test node text", metadata={"key": "value"})]
    index = VectorStoreIndex(
        input_nodes,
        storage_context=storage_context,
        service_context=mock_service_context,
    )

    retriever = index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve("What is?")
    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "test node text"
    assert nodes[0].node.metadata == {"key": "value"}
    deeplake.delete(dataset_path)
