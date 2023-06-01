"""Test list index."""

from typing import Dict, List, Tuple

from llama_index.data_structs.node import Node
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.list.base import GPTListIndex, ListRetrieverMode
from llama_index.indices.service_context import ServiceContext
from llama_index.readers.schema.base import Document


def test_build_list(
    documents: List[Document], mock_service_context: ServiceContext
) -> None:
    """Test build list."""
    list_index = GPTListIndex.from_documents(
        documents, service_context=mock_service_context
    )
    assert len(list_index.index_struct.nodes) == 4
    # check contents of nodes
    node_ids = list_index.index_struct.nodes
    nodes = list_index.docstore.get_nodes(node_ids)
    assert nodes[0].text == "Hello world."
    assert nodes[1].text == "This is a test."
    assert nodes[2].text == "This is another test."
    assert nodes[3].text == "This is a test v2."


def test_refresh_list(
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test build list."""
    # add extra document
    more_documents = documents + [Document("Test document 2")]

    # ensure documents have doc_id
    for i in range(len(more_documents)):
        more_documents[i].doc_id = str(i)

    # create index
    list_index = GPTListIndex.from_documents(
        more_documents, service_context=mock_service_context
    )

    # check that no documents are refreshed
    refreshed_docs = list_index.refresh_ref_docs(more_documents)
    assert refreshed_docs[0] is False
    assert refreshed_docs[1] is False

    # modify a document and test again
    more_documents = documents + [Document("Test document 2, now with changes!")]
    for i in range(len(more_documents)):
        more_documents[i].doc_id = str(i)

    # second document should refresh
    refreshed_docs = list_index.refresh_ref_docs(more_documents)
    assert refreshed_docs[0] is False
    assert refreshed_docs[1] is True

    test_node = list_index.docstore.get_node(list_index.index_struct.nodes[-1])
    assert test_node.text == "Test document 2, now with changes!"


def test_build_list_multiple(mock_service_context: ServiceContext) -> None:
    """Test build list multiple."""
    documents = [
        Document("Hello world.\nThis is a test."),
        Document("This is another test.\nThis is a test v2."),
    ]
    list_index = GPTListIndex.from_documents(
        documents, service_context=mock_service_context
    )
    assert len(list_index.index_struct.nodes) == 4
    nodes = list_index.docstore.get_nodes(list_index.index_struct.nodes)
    # check contents of nodes
    assert nodes[0].text == "Hello world."
    assert nodes[1].text == "This is a test."
    assert nodes[2].text == "This is another test."
    assert nodes[3].text == "This is a test v2."


def test_list_insert(
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test insert to list."""
    list_index = GPTListIndex([], service_context=mock_service_context)
    assert len(list_index.index_struct.nodes) == 0
    list_index.insert(documents[0])
    nodes = list_index.docstore.get_nodes(list_index.index_struct.nodes)
    # check contents of nodes
    assert nodes[0].text == "Hello world."
    assert nodes[1].text == "This is a test."
    assert nodes[2].text == "This is another test."
    assert nodes[3].text == "This is a test v2."

    # test insert with ID
    document = documents[0]
    document.doc_id = "test_id"
    list_index = GPTListIndex([])
    list_index.insert(document)
    # check contents of nodes
    nodes = list_index.docstore.get_nodes(list_index.index_struct.nodes)
    # check contents of nodes
    for node in nodes:
        assert node.ref_doc_id == "test_id"


def test_list_delete(
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test insert to list and then delete."""
    new_documents = [
        Document("Hello world.\nThis is a test.", doc_id="test_id_1"),
        Document("This is another test.", doc_id="test_id_2"),
        Document("This is a test v2.", doc_id="test_id_3"),
    ]

    list_index = GPTListIndex.from_documents(
        new_documents, service_context=mock_service_context
    )

    # test ref doc info for three docs
    all_ref_doc_info = list_index.ref_doc_info
    for idx, ref_doc_id in enumerate(all_ref_doc_info.keys()):
        assert new_documents[idx].doc_id == ref_doc_id

    # delete from documents
    list_index.delete_ref_doc("test_id_1")
    assert len(list_index.index_struct.nodes) == 2
    nodes = list_index.docstore.get_nodes(list_index.index_struct.nodes)
    assert nodes[0].ref_doc_id == "test_id_2"
    assert nodes[0].text == "This is another test."
    assert nodes[1].ref_doc_id == "test_id_3"
    assert nodes[1].text == "This is a test v2."
    # check that not in docstore anymore
    source_doc = list_index.docstore.get_document("test_id_1", raise_error=False)
    assert source_doc is None

    list_index = GPTListIndex.from_documents(
        new_documents, service_context=mock_service_context
    )
    list_index.delete_ref_doc("test_id_2")
    assert len(list_index.index_struct.nodes) == 3
    nodes = list_index.docstore.get_nodes(list_index.index_struct.nodes)
    assert nodes[0].ref_doc_id == "test_id_1"
    assert nodes[0].text == "Hello world."
    assert nodes[1].ref_doc_id == "test_id_1"
    assert nodes[1].text == "This is a test."
    assert nodes[2].ref_doc_id == "test_id_3"
    assert nodes[2].text == "This is a test v2."


def _get_embeddings(
    query_str: str, nodes: List[Node]
) -> Tuple[List[float], List[List[float]]]:
    """Get node text embedding similarity."""
    text_embed_map: Dict[str, List[float]] = {
        "Hello world.": [1.0, 0.0, 0.0, 0.0, 0.0],
        "This is a test.": [0.0, 1.0, 0.0, 0.0, 0.0],
        "This is another test.": [0.0, 0.0, 1.0, 0.0, 0.0],
        "This is a test v2.": [0.0, 0.0, 0.0, 1.0, 0.0],
    }
    node_embeddings = []
    for node in nodes:
        node_embeddings.append(text_embed_map[node.get_text()])

    return [1.0, 0, 0, 0, 0], node_embeddings


def test_as_retriever(
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    list_index = GPTListIndex.from_documents(
        documents, service_context=mock_service_context
    )
    default_retriever = list_index.as_retriever(
        retriever_mode=ListRetrieverMode.DEFAULT
    )
    assert isinstance(default_retriever, BaseRetriever)

    embedding_retriever = list_index.as_retriever(
        retriever_mode=ListRetrieverMode.EMBEDDING
    )
    assert isinstance(embedding_retriever, BaseRetriever)
