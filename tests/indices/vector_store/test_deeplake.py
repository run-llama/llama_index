"""Test deeplake indexes."""

import os
from typing import List

import pytest

from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.types import NodeWithEmbedding
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.schema import Document
from llama_index.schema import TextNode
from llama_index.storage.storage_context import StorageContext


EMBEDDING_DIM = 100
NUMBER_OF_DATA = 10


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

    node = nodes[0].node

    result = NodeWithEmbedding(
        node=node,
        embedding=[1.0 for i in range(EMBEDDING_DIM)],
    )
    results = [result for i in range(NUMBER_OF_DATA)]
    vector_store.add(results)
    assert len(vector_store.vectorstore) == 14

    ref_doc_id = str(node.ref_doc_id)
    vector_store.delete(ref_doc_id)
    assert len(vector_store.vectorstore) == 3
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


@pytest.mark.skipif("CI" in os.environ, reason="no DeepLake in CI")
def test_backwards_compatibility() -> None:
    import deeplake
    from deeplake.core.vectorstore import utils

    # create data
    texts, embeddings, ids, metadatas, images = utils.create_data(
        number_of_data=NUMBER_OF_DATA, embedding_dim=EMBEDDING_DIM
    )
    metadatas = [metadata.update({"doc_id": "2"}) for metadata in metadatas]
    node = TextNode(
        text="test node text", metadata={"key": "value", "doc_id": "1"}, id_="1"
    )
    result = NodeWithEmbedding(
        node=node,
        embedding=[1.0 for i in range(EMBEDDING_DIM)],
    )

    results = [result for i in range(10)]

    dataset_path = "local_ds1"
    ds = deeplake.empty(dataset_path)
    ds.create_tensor("ids", htype="text")
    ds.create_tensor("embedding", htype="embedding")
    ds.create_tensor("text", htype="text")
    ds.create_tensor("metadata", htype="json")

    ds.extend(
        {
            "ids": ids,
            "text": texts,
            "metadata": metadatas,
            "embedding": embeddings,
        }
    )

    vectorstore = DeepLakeVectorStore(
        dataset_path=dataset_path,
        overwrite=False,
        verbose=False,
    )

    vectorstore.add(results)
    assert len(vectorstore.vectorstore) == 20
    deeplake.delete(dataset_path)
