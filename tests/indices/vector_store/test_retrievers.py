from typing import List, cast
from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import GPTVectorStoreIndex
from llama_index.readers.schema.base import Document
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.simple import SimpleVectorStore


def test_faiss_query(
    documents: List[Document],
    faiss_storage_context: StorageContext,
    mock_service_context: ServiceContext,
) -> None:
    """Test embedding query."""
    index = GPTVectorStoreIndex.from_documents(
        documents=documents,
        storage_context=faiss_storage_context,
        service_context=mock_service_context,
    )

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) == 1
    assert nodes[0].node.text == "This is another test."


def test_simple_query(
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test embedding query."""
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=mock_service_context
    )

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) == 1
    assert nodes[0].node.text == "This is another test."


def test_query_and_similarity_scores(
    mock_service_context: ServiceContext,
) -> None:
    """Test that sources nodes have similarity scores."""
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    document = Document(doc_text)
    index = GPTVectorStoreIndex.from_documents(
        [document], service_context=mock_service_context
    )

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) > 0
    assert nodes[0].score is not None


def test_simple_check_ids(
    mock_service_context: ServiceContext,
) -> None:
    """Test build GPTVectorStoreIndex."""
    ref_doc_id = "ref_doc_id_test"
    source_rel = {DocumentRelationship.SOURCE: ref_doc_id}
    all_nodes = [
        Node("Hello world.", doc_id="node1", relationships=source_rel),
        Node("This is a test.", doc_id="node2", relationships=source_rel),
        Node("This is another test.", doc_id="node3", relationships=source_rel),
        Node("This is a test v2.", doc_id="node4", relationships=source_rel),
    ]
    index = GPTVectorStoreIndex(all_nodes, service_context=mock_service_context)

    # test query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert nodes[0].node.text == "This is another test."
    assert nodes[0].node.ref_doc_id == "ref_doc_id_test"
    assert nodes[0].node.doc_id == "node3"
    vector_store = cast(SimpleVectorStore, index._vector_store)
    assert "node3" in vector_store._data.embedding_dict
    assert "node3" in vector_store._data.text_id_to_ref_doc_id


def test_faiss_check_ids(
    mock_service_context: ServiceContext,
    faiss_storage_context: StorageContext,
) -> None:
    """Test embedding query."""

    ref_doc_id = "ref_doc_id_test"
    source_rel = {DocumentRelationship.SOURCE: ref_doc_id}
    all_nodes = [
        Node("Hello world.", doc_id="node1", relationships=source_rel),
        Node("This is a test.", doc_id="node2", relationships=source_rel),
        Node("This is another test.", doc_id="node3", relationships=source_rel),
        Node("This is a test v2.", doc_id="node4", relationships=source_rel),
    ]

    index = GPTVectorStoreIndex(
        all_nodes,
        storage_context=faiss_storage_context,
        service_context=mock_service_context,
    )

    # test query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert nodes[0].node.text == "This is another test."
    assert nodes[0].node.ref_doc_id == "ref_doc_id_test"
    assert nodes[0].node.doc_id == "node3"


def test_query_and_count_tokens(mock_service_context: ServiceContext) -> None:
    """Test embedding query."""
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    document = Document(doc_text)
    index = GPTVectorStoreIndex.from_documents(
        [document], service_context=mock_service_context
    )
    assert index.service_context.embed_model.total_tokens_used == 20

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever()
    _ = retriever.retrieve(QueryBundle(query_str))
    assert index.service_context.embed_model.last_token_usage == 3
