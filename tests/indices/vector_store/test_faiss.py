"""Test vector store indexes."""

from pathlib import Path
from typing import List

import pytest

from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.schema import Document, TextNode
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.types import NodeWithEmbedding, VectorStoreQuery

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
def test_build_faiss(
    documents: List[Document],
    faiss_storage_context: StorageContext,
    mock_service_context: ServiceContext,
) -> None:
    """Test build VectorStoreIndex with FaissVectoreStore."""
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=faiss_storage_context,
        service_context=mock_service_context,
    )
    assert len(index.index_struct.nodes_dict) == 4

    node_ids = list(index.index_struct.nodes_dict.values())
    nodes = index.docstore.get_nodes(node_ids)
    node_texts = [node.get_content() for node in nodes]
    assert "Hello world." in node_texts
    assert "This is a test." in node_texts
    assert "This is another test." in node_texts
    assert "This is a test v2." in node_texts


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
def test_faiss_insert(
    documents: List[Document],
    faiss_storage_context: StorageContext,
    mock_service_context: ServiceContext,
) -> None:
    """Test insert VectorStoreIndex with FaissVectoreStore."""
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=faiss_storage_context,
        service_context=mock_service_context,
    )

    # insert into index
    index.insert(Document(text="This is a test v3."))

    # check contents of nodes
    node_ids = list(index.index_struct.nodes_dict.values())
    nodes = index.docstore.get_nodes(node_ids)
    node_texts = [node.get_content() for node in nodes]
    assert "This is a test v2." in node_texts
    assert "This is a test v3." in node_texts


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
def test_persist(tmp_path: Path) -> None:
    import faiss

    vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(5))

    vector_store.add(
        [
            NodeWithEmbedding(
                node=TextNode(text="test text"),
                embedding=[0, 0, 0, 1, 1],
            )
        ]
    )

    result = vector_store.query(VectorStoreQuery(query_embedding=[0, 0, 0, 1, 1]))

    persist_path = str(tmp_path / "faiss.index")
    vector_store.persist(persist_path)
    new_vector_store = FaissVectorStore.from_persist_path(persist_path)
    new_result = new_vector_store.query(
        VectorStoreQuery(query_embedding=[0, 0, 0, 1, 1])
    )

    assert result == new_result
