"""Test document summary index."""

from typing import Any, List

import pytest
from llama_index.core.data_structs.document_summary import IndexDocumentSummary
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


@pytest.mark.asyncio
async def test_adelete_nodes(
    index: DocumentSummaryIndex,
) -> None:
    """Test asynchronously deleting nodes."""
    nodes = list(index.index_struct.node_id_to_summary_id.keys())
    await index.adelete_nodes([nodes[0], nodes[1]])

    assert len(index.ref_doc_info) == 2
    assert len(index.index_struct.doc_id_to_summary_id) == 2
    assert len(index.index_struct.node_id_to_summary_id) == 2
    assert len(index.index_struct.summary_id_to_node_ids) == 2
    assert len(index.vector_store._data.embedding_dict) == 2  # type: ignore

    results = index.as_retriever(similarity_top_k=4).retrieve("Test query")
    assert {result.node.node_id for result in results} == set(nodes[2:])


@pytest.mark.asyncio
async def test_adelete_nodes_uses_async_vector_store_delete(
    index: DocumentSummaryIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test async deletion does not enter the synchronous event-loop adapter."""
    node_id = next(iter(index.index_struct.node_id_to_summary_id))
    ref_doc_id = index.docstore.get_node(node_id).ref_doc_id
    assert ref_doc_id is not None

    sync_delete = index.vector_store.delete
    async_deletes: List[str] = []

    def fail_sync_delete(self: Any, ref_doc_id: str, **kwargs: Any) -> None:
        raise AssertionError("Synchronous vector deletion must not be called")

    async def record_async_delete(self: Any, ref_doc_id: str, **kwargs: Any) -> None:
        async_deletes.append(ref_doc_id)
        sync_delete(ref_doc_id, **kwargs)

    vector_store_type = type(index.vector_store)
    monkeypatch.setattr(vector_store_type, "delete", fail_sync_delete)
    monkeypatch.setattr(vector_store_type, "adelete", record_async_delete)

    await index.adelete_nodes([node_id])

    assert async_deletes == [ref_doc_id]


@pytest.mark.asyncio
async def test_adelete_nodes_persists_partial_deletion(
    index: DocumentSummaryIndex,
) -> None:
    """Test partial node deletion is written back to the index store."""
    node_id = next(iter(index.index_struct.node_id_to_summary_id))
    summary_id = index.index_struct.node_id_to_summary_id[node_id]
    extra_node = index.docstore.get_node(node_id).model_copy(
        update={"id_": "extra_node"}
    )
    index.docstore.add_documents([extra_node])
    index.index_struct.summary_id_to_node_ids[summary_id].append(extra_node.node_id)
    index.index_struct.node_id_to_summary_id[extra_node.node_id] = summary_id
    await index.storage_context.index_store.async_add_index_struct(index.index_struct)

    await index.adelete_nodes([extra_node.node_id])

    persisted = await index.storage_context.index_store.aget_index_struct(
        index.index_struct.index_id
    )
    assert isinstance(persisted, IndexDocumentSummary)
    assert extra_node.node_id not in persisted.node_id_to_summary_id
    assert persisted.node_id_to_summary_id[node_id] == summary_id
    assert persisted.summary_id_to_node_ids[summary_id] == [node_id]


@pytest.mark.asyncio
async def test_adelete_nodes_deletes_from_docstore(
    index: DocumentSummaryIndex,
) -> None:
    """Test async deletion honors delete_from_docstore for an emptied document."""
    node_id = next(iter(index.index_struct.node_id_to_summary_id))
    summary_id = index.index_struct.node_id_to_summary_id[node_id]
    ref_doc_id = index.docstore.get_node(node_id).ref_doc_id
    assert ref_doc_id is not None

    await index.adelete_nodes([node_id], delete_from_docstore=True)

    assert await index.docstore.aget_node(node_id, raise_error=False) is None
    assert await index.docstore.aget_node(summary_id, raise_error=False) is None
    assert await index.docstore.aget_ref_doc_info(ref_doc_id) is None


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
