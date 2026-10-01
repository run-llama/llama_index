"""Test document summary index."""

from typing import List

import pytest
from llama_index.core.indices.document_summary.base import DocumentSummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import Document
from tests.mock_utils.mock_prompts import MOCK_REFINE_PROMPT, MOCK_TEXT_QA_PROMPT


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


def test_delete_nodes_ignores_invalid_node_ids(
    index: DocumentSummaryIndex,
) -> None:
    before_node_to_summary = dict(index.index_struct.node_id_to_summary_id)
    before_summary_to_nodes = {
        summary_id: list(node_ids)
        for summary_id, node_ids in index.index_struct.summary_id_to_node_ids.items()
    }

    index.delete_nodes(["does_not_exist_1", "does_not_exist_2"])

    assert dict(index.index_struct.node_id_to_summary_id) == before_node_to_summary
    assert {
        summary_id: list(node_ids)
        for summary_id, node_ids in index.index_struct.summary_id_to_node_ids.items()
    } == before_summary_to_nodes


def test_delete_nodes_deletes_valid_ids_and_skips_invalid_ones(
    index: DocumentSummaryIndex,
) -> None:
    nodes = list(index.index_struct.node_id_to_summary_id.keys())
    valid_node_id = nodes[0]

    index.delete_nodes(["does_not_exist_1", valid_node_id, "does_not_exist_2"])

    assert valid_node_id not in index.index_struct.node_id_to_summary_id


def test_delete_nodes_keeps_docstore_by_default(
    docs: List[Document],
    index: DocumentSummaryIndex,
) -> None:
    """Without the flag the docstore is left untouched."""
    node_ids = list(index.index_struct.node_id_to_summary_id.keys())
    before = set(index.docstore.docs.keys())

    index.delete_nodes([node_ids[0], node_ids[1]])

    assert set(index.docstore.docs.keys()) == before


def test_delete_nodes_from_docstore(
    docs: List[Document],
    index: DocumentSummaryIndex,
) -> None:
    """`delete_from_docstore=True` must remove the nodes and their emptied docs."""
    node_ids = list(index.index_struct.node_id_to_summary_id.keys())
    victims = [node_ids[0], node_ids[1]]
    # in this fixture each document contributes exactly one indexed node, so
    # deleting these empties doc_1 and doc_2 entirely
    doomed_docs = ["doc_1", "doc_2"]
    survivors = ["doc_3", "doc_4"]

    index.delete_nodes(victims, delete_from_docstore=True)

    remaining = set(index.docstore.docs.keys())
    for node_id in victims:
        assert node_id not in remaining
    for doc_id in doomed_docs:
        assert index.docstore.get_ref_doc_info(doc_id) is None
    # documents that still have nodes are untouched
    for doc_id in survivors:
        assert index.docstore.get_ref_doc_info(doc_id) is not None


def test_delete_nodes_from_docstore_partial_document(
    patch_llm_predictor,
    mock_embed_model,
) -> None:
    """
    Deleting some of a document's nodes removes only those nodes.

    The document keeps its remaining nodes and is not deleted.
    """
    long_doc = Document(
        text=". ".join(f"sentence number {i}" for i in range(40)), id_="big"
    )
    index = DocumentSummaryIndex.from_documents(
        [long_doc],
        response_synthesizer=get_response_synthesizer(
            text_qa_template=MOCK_TEXT_QA_PROMPT,
            refine_template=MOCK_REFINE_PROMPT,
        ),
        summary_query="summary_query",
        embed_model=mock_embed_model,
        transformations=[SentenceSplitter(chunk_size=32, chunk_overlap=0)],
    )
    node_ids = list(index.index_struct.node_id_to_summary_id.keys())
    assert len(node_ids) > 2, "fixture must produce several nodes for one document"

    victims, keepers = node_ids[:2], node_ids[2:]
    index.delete_nodes(victims, delete_from_docstore=True)

    remaining = set(index.docstore.docs.keys())
    for node_id in victims:
        assert node_id not in remaining
    for node_id in keepers:
        assert node_id in remaining
    # the document itself survives because it still has nodes
    assert index.docstore.get_ref_doc_info("big") is not None
