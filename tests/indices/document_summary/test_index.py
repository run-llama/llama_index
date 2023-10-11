"""Test document summary index."""

from typing import List

from llama_index.indices.document_summary.base import DocumentSummaryIndex
from llama_index.schema import Document


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
