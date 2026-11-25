from pathlib import Path
from typing import List
import pytest

from llama_index.core.indices.composability.graph import ComposableGraph
from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.indices.loading import load_graph_from_storage
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.storage.storage_context import StorageContext


def test_load_graph_from_storage_simple(
    documents: List[Document], tmp_path: Path
) -> None:
    # construct simple (i.e. in memory) storage context
    storage_context = StorageContext.from_defaults()

    # construct index
    vector_index_1 = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
    )

    # construct second index, testing vector store overlap
    vector_index_2 = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
    )

    # construct index
    summary_index = SummaryIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
    )

    # construct graph
    graph = ComposableGraph.from_indices(
        SummaryIndex,
        children_indices=[vector_index_1, vector_index_2, summary_index],
        index_summaries=["vector index 1", "vector index 2", "summary index"],
        storage_context=storage_context,
    )

    query_engine = graph.as_query_engine()
    response = query_engine.query("test query")

    # persist storage to disk
    storage_context.persist(str(tmp_path))

    # load storage context
    new_storage_context = StorageContext.from_defaults(persist_dir=str(tmp_path))

    # load index
    new_graph = load_graph_from_storage(new_storage_context, root_id=graph.root_id)

    assert (
        new_graph.get_index(vector_index_1.index_id).index_struct.summary
        == "vector index 1"
    )
    assert (
        new_graph.get_index(vector_index_2.index_id).index_struct.summary
        == "vector index 2"
    )
    assert (
        new_graph.get_index(summary_index.index_id).index_struct.summary
        == "summary index"
    )

    new_query_engine = new_graph.as_query_engine()
    new_response = new_query_engine.query("test query")

    assert str(response) == str(new_response)


def test_composable_graph_mismatched_summaries_does_not_mutate_indices(
    documents: List[Document],
) -> None:
    storage_context = StorageContext.from_defaults()
    index_1 = SummaryIndex.from_documents(documents=documents, storage_context=storage_context)
    index_2 = SummaryIndex.from_documents(documents=documents, storage_context=storage_context)

    with pytest.raises(ValueError, match="indices and index_summaries must have same length!"):
        ComposableGraph.from_indices(
            SummaryIndex,
            children_indices=[index_1, index_2],
            index_summaries=["only one summary"],
            storage_context=storage_context,
        )

    assert index_1.index_struct.summary is None
    assert index_2.index_struct.summary is None
