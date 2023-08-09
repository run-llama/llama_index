from typing import List, cast

from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.indices.vector_store.sentence_window import SentenceWindowVectorIndex
from llama_index.schema import Document, MetadataMode
from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.simple import SimpleVectorStore


def test_faiss_query(
    documents: List[Document],
    faiss_storage_context: StorageContext,
    mock_service_context: ServiceContext,
) -> None:
    """Test embedding query."""
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=faiss_storage_context,
        service_context=mock_service_context,
    )

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "This is another test."


def test_simple_query(
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test embedding query."""
    index = VectorStoreIndex.from_documents(
        documents, service_context=mock_service_context
    )

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "This is another test."


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
    document = Document(text=doc_text)
    index = VectorStoreIndex.from_documents(
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
    """Test build VectorStoreIndex."""
    ref_doc_id = "ref_doc_id_test"
    source_rel = {NodeRelationship.SOURCE: RelatedNodeInfo(node_id=ref_doc_id)}
    all_nodes = [
        TextNode(text="Hello world.", id_="node1", relationships=source_rel),
        TextNode(text="This is a test.", id_="node2", relationships=source_rel),
        TextNode(text="This is another test.", id_="node3", relationships=source_rel),
        TextNode(text="This is a test v2.", id_="node4", relationships=source_rel),
    ]
    index = VectorStoreIndex(all_nodes, service_context=mock_service_context)

    # test query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert nodes[0].node.get_content() == "This is another test."
    assert nodes[0].node.ref_doc_id == "ref_doc_id_test"
    assert nodes[0].node.node_id == "node3"
    vector_store = cast(SimpleVectorStore, index._vector_store)
    assert "node3" in vector_store._data.embedding_dict
    assert "node3" in vector_store._data.text_id_to_ref_doc_id


def test_faiss_check_ids(
    mock_service_context: ServiceContext,
    faiss_storage_context: StorageContext,
) -> None:
    """Test embedding query."""

    ref_doc_id = "ref_doc_id_test"
    source_rel = {NodeRelationship.SOURCE: RelatedNodeInfo(node_id=ref_doc_id)}
    all_nodes = [
        TextNode(text="Hello world.", id_="node1", relationships=source_rel),
        TextNode(text="This is a test.", id_="node2", relationships=source_rel),
        TextNode(text="This is another test.", id_="node3", relationships=source_rel),
        TextNode(text="This is a test v2.", id_="node4", relationships=source_rel),
    ]

    index = VectorStoreIndex(
        all_nodes,
        storage_context=faiss_storage_context,
        service_context=mock_service_context,
    )

    # test query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert nodes[0].node.get_content() == "This is another test."
    assert nodes[0].node.ref_doc_id == "ref_doc_id_test"
    assert nodes[0].node.node_id == "node3"


def test_query(mock_service_context: ServiceContext) -> None:
    """Test embedding query."""
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    document = Document(text=doc_text)
    index = VectorStoreIndex.from_documents(
        [document], service_context=mock_service_context
    )

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever()
    _ = retriever.retrieve(QueryBundle(query_str))


def test_sentence_window_query(mock_service_context: ServiceContext) -> None:
    """Text sentence window query."""
    doc_text = "Hello world. This is a test. This is another test. This is a test v2."
    document = Document(text=doc_text)
    index = SentenceWindowVectorIndex.from_documents(
        [document], service_context=mock_service_context
    )

    # test sentence window query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) == 2
    assert nodes[0].node.get_content(metadata_mode=MetadataMode.NONE) == doc_text
    assert nodes[0].node.metadata["original_text"] == "This is another test."
    assert nodes[0].node.metadata["window"] == doc_text
