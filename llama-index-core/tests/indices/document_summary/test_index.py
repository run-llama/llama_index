"""Test document summary index."""

from typing import List

import pytest
from llama_index.core.indices.document_summary.base import DocumentSummaryIndex
from llama_index.core.schema import Document


def test_build_index(
    docs: List[Document],
    index: DocumentSummaryIndex,
) -> None:
    """Test build tree."""
    test = index.get_document_summary("doc_1")
    assert test == "summary_query:This is a test v2."
    test4 = index.get_document_summary("doc_4")
    assert test4 == "summary_query:Hello world."

    all_ref_doc_info = index.ref_doc_info
    for idx, (doc_id, ref_doc_info) in enumerate(all_ref_doc_info.items()):
        assert docs[idx].doc_id == doc_id
        assert len(ref_doc_info.node_ids) == 2


def test_delete_ref_doc(
    docs: List[Document],
    index: DocumentSummaryIndex,
) -> None:
    """Test delete node."""
    index.delete_ref_doc("doc_1")

    # assert that error is raised for doc_1
    with pytest.raises(ValueError):
        index.get_document_summary("doc_1")

    assert index.get_document_summary("doc_2") == "summary_query:This is another test."
    assert index.get_document_summary("doc_3") == "summary_query:This is a test."
    assert index.get_document_summary("doc_4") == "summary_query:Hello world."

    assert len(index.ref_doc_info) == 3
    assert len(index.index_struct.doc_id_to_summary_id) == 3
    assert len(index.index_struct.node_id_to_summary_id) == 3
    assert len(index.index_struct.summary_id_to_node_ids) == 3

    assert len(index.vector_store._data.embedding_dict) == 3  # type: ignore


def test_delete_nodes(
    docs: List[Document],
    index: DocumentSummaryIndex,
) -> None:
    """Test delete node."""
    nodes = list(index.index_struct.node_id_to_summary_id.keys())
    index.delete_nodes([nodes[0], nodes[1]])

    assert len(index.ref_doc_info) == 2
    assert len(index.index_struct.doc_id_to_summary_id) == 2
    assert len(index.index_struct.node_id_to_summary_id) == 2
    assert len(index.index_struct.summary_id_to_node_ids) == 2

    assert len(index.vector_store._data.embedding_dict) == 2  # type: ignore
