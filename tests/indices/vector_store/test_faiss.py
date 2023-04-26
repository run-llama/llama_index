"""Test vector store indexes."""

from typing import List

from gpt_index.indices.service_context import ServiceContext
from gpt_index.indices.vector_store.base import GPTVectorStoreIndex

from gpt_index.readers.schema.base import Document
from gpt_index.storage.storage_context import StorageContext


def test_build_faiss(
    documents: List[Document],
    faiss_storage_context: StorageContext,
    mock_service_context: ServiceContext,
) -> None:
    """Test build GPTFaissIndex."""
    index = GPTVectorStoreIndex.from_documents(
        documents=documents,
        storage_context=faiss_storage_context,
        service_context=mock_service_context,
    )
    assert len(index.index_struct.nodes_dict) == 4

    node_ids = list(index.index_struct.nodes_dict.values())
    nodes = index.docstore.get_nodes(node_ids)
    node_texts = [node.text for node in nodes]
    assert "Hello world." in node_texts
    assert "This is a test." in node_texts
    assert "This is another test." in node_texts
    assert "This is a test v2." in node_texts


def test_faiss_insert(
    documents: List[Document],
    faiss_storage_context: StorageContext,
    mock_service_context: ServiceContext,
) -> None:
    """Test build GPTFaissIndex."""
    index = GPTVectorStoreIndex.from_documents(
        documents=documents,
        storage_context=faiss_storage_context,
        service_context=mock_service_context,
    )

    node_ids = index.index_struct.nodes_dict
    print(node_ids)

    # insert into index
    index.insert(Document(text="This is a test v3."))

    # check contents of nodes
    node_ids = index.index_struct.nodes_dict
    print(node_ids)

    node_ids = list(index.index_struct.nodes_dict.values())
    nodes = index.docstore.get_nodes(node_ids)
    print("nodes")
    print(nodes)
    node_texts = [node.text for node in nodes]
    assert "This is a test v2." in node_texts
    assert "This is a test v3." in node_texts
