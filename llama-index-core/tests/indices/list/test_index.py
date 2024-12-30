"""Test summary index."""

from typing import List

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.list.base import ListRetrieverMode, SummaryIndex
from llama_index.core.schema import Document


def test_build_list(documents: List[Document], patch_token_text_splitter) -> None:
    """Test build list."""
    summary_index = SummaryIndex.from_documents(documents)
    assert len(summary_index.index_struct.nodes) == 4
    # check contents of nodes
    node_ids = summary_index.index_struct.nodes
    nodes = summary_index.docstore.get_nodes(node_ids)
    assert nodes[0].get_content() == "Hello world."
    assert nodes[1].get_content() == "This is a test."
    assert nodes[2].get_content() == "This is another test."
    assert nodes[3].get_content() == "This is a test v2."


def test_refresh_list(documents: List[Document]) -> None:
    """Test build list."""
    # add extra document
    more_documents = [*documents, Document(text="Test document 2")]

    # ensure documents have doc_id
    for i in range(len(more_documents)):
        more_documents[i].doc_id = str(i)  # type: ignore[misc]

    # create index
    summary_index = SummaryIndex.from_documents(more_documents)

    # check that no documents are refreshed
    refreshed_docs = summary_index.refresh_ref_docs(more_documents)
    assert refreshed_docs[0] is False
    assert refreshed_docs[1] is False

    # modify a document and test again
    more_documents = [*documents, Document(text="Test document 2, now with changes!")]
    for i in range(len(more_documents)):
        more_documents[i].doc_id = str(i)  # type: ignore[misc]

    # second document should refresh
    refreshed_docs = summary_index.refresh_ref_docs(more_documents)
    assert refreshed_docs[0] is False
    assert refreshed_docs[1] is True

    test_node = summary_index.docstore.get_node(summary_index.index_struct.nodes[-1])
    assert test_node.get_content() == "Test document 2, now with changes!"


def test_build_list_multiple(patch_token_text_splitter) -> None:
    """Test build list multiple."""
    documents = [
        Document(text="Hello world.\nThis is a test."),
        Document(text="This is another test.\nThis is a test v2."),
    ]
    summary_index = SummaryIndex.from_documents(documents)
    assert len(summary_index.index_struct.nodes) == 4
    nodes = summary_index.docstore.get_nodes(summary_index.index_struct.nodes)
    # check contents of nodes
    assert nodes[0].get_content() == "Hello world."
    assert nodes[1].get_content() == "This is a test."
    assert nodes[2].get_content() == "This is another test."
    assert nodes[3].get_content() == "This is a test v2."


def test_list_insert(documents: List[Document], patch_token_text_splitter) -> None:
    """Test insert to list."""
    summary_index = SummaryIndex([])
    assert len(summary_index.index_struct.nodes) == 0
    summary_index.insert(documents[0])
    nodes = summary_index.docstore.get_nodes(summary_index.index_struct.nodes)
    # check contents of nodes
    assert nodes[0].get_content() == "Hello world."
    assert nodes[1].get_content() == "This is a test."
    assert nodes[2].get_content() == "This is another test."
    assert nodes[3].get_content() == "This is a test v2."

    # test insert with ID
    document = documents[0]
    document.doc_id = "test_id"  # type: ignore[misc]
    summary_index = SummaryIndex([])
    summary_index.insert(document)
    # check contents of nodes
    nodes = summary_index.docstore.get_nodes(summary_index.index_struct.nodes)
    # check contents of nodes
    for node in nodes:
        assert node.ref_doc_id == "test_id"


def test_list_delete(documents: List[Document], patch_token_text_splitter) -> None:
    """Test insert to list and then delete."""
    new_documents = [
        Document(text="Hello world.\nThis is a test.", id_="test_id_1"),
        Document(text="This is another test.", id_="test_id_2"),
        Document(text="This is a test v2.", id_="test_id_3"),
    ]

    summary_index = SummaryIndex.from_documents(new_documents)

    # test ref doc info for three docs
    all_ref_doc_info = summary_index.ref_doc_info
    for idx, ref_doc_id in enumerate(all_ref_doc_info.keys()):
        assert new_documents[idx].doc_id == ref_doc_id

    # delete from documents
    summary_index.delete_ref_doc("test_id_1")
    assert len(summary_index.index_struct.nodes) == 2
    nodes = summary_index.docstore.get_nodes(summary_index.index_struct.nodes)
    assert nodes[0].ref_doc_id == "test_id_2"
    assert nodes[0].get_content() == "This is another test."
    assert nodes[1].ref_doc_id == "test_id_3"
    assert nodes[1].get_content() == "This is a test v2."
    # check that not in docstore anymore
    source_doc = summary_index.docstore.get_document("test_id_1", raise_error=False)
    assert source_doc is None

    summary_index = SummaryIndex.from_documents(new_documents)
    summary_index.delete_ref_doc("test_id_2")
    assert len(summary_index.index_struct.nodes) == 3
    nodes = summary_index.docstore.get_nodes(summary_index.index_struct.nodes)
    assert nodes[0].ref_doc_id == "test_id_1"
    assert nodes[0].get_content() == "Hello world."
    assert nodes[1].ref_doc_id == "test_id_1"
    assert nodes[1].get_content() == "This is a test."
    assert nodes[2].ref_doc_id == "test_id_3"
    assert nodes[2].get_content() == "This is a test v2."


def test_as_retriever(documents: List[Document]) -> None:
    summary_index = SummaryIndex.from_documents(documents)
    default_retriever = summary_index.as_retriever(
        retriever_mode=ListRetrieverMode.DEFAULT
    )
    assert isinstance(default_retriever, BaseRetriever)

    embedding_retriever = summary_index.as_retriever(
        retriever_mode=ListRetrieverMode.EMBEDDING
    )
    assert isinstance(embedding_retriever, BaseRetriever)
