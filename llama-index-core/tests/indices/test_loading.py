from pathlib import Path
from typing import List

import pytest
from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.indices.loading import (
    load_index_from_storage,
    load_indices_from_storage,
)
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import BaseNode, Document
from llama_index.core.storage.storage_context import StorageContext


def test_load_index_from_storage_simple(
    documents: List[Document], tmp_path: Path
) -> None:
    # construct simple (i.e. in memory) storage context
    storage_context = StorageContext.from_defaults()

    # construct index
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
    )

    # persist storage to disk
    storage_context.persist(str(tmp_path))

    # load storage context
    new_storage_context = StorageContext.from_defaults(persist_dir=str(tmp_path))

    # load index
    new_index = load_index_from_storage(storage_context=new_storage_context)

    assert index.index_id == new_index.index_id


def test_load_index_from_storage_multiple(
    nodes: List[BaseNode], tmp_path: Path
) -> None:
    # construct simple (i.e. in memory) storage context
    storage_context = StorageContext.from_defaults()

    # add nodes to docstore
    storage_context.docstore.add_documents(nodes)

    # construct multiple indices
    vector_index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    vector_id = vector_index.index_id

    summary_index = SummaryIndex(nodes=nodes, storage_context=storage_context)

    list_id = summary_index.index_id

    # persist storage to disk
    storage_context.persist(str(tmp_path))

    # load storage context
    new_storage_context = StorageContext.from_defaults(persist_dir=str(tmp_path))

    # load single index should fail since there are multiple indices in index store
    with pytest.raises(ValueError):
        load_index_from_storage(new_storage_context)

    # test load all indices
    indices = load_indices_from_storage(storage_context)
    index_ids = [index.index_id for index in indices]
    assert len(index_ids) == 2
    assert vector_id in index_ids
    assert list_id in index_ids

    # test load multiple indices by ids
    indices = load_indices_from_storage(storage_context, index_ids=[list_id, vector_id])
    index_ids = [index.index_id for index in indices]
    assert len(index_ids) == 2
    assert vector_id in index_ids
    assert list_id in index_ids


def test_load_index_from_storage_retrieval_result_identical(
    documents: List[Document], tmp_path: Path
) -> None:
    # construct simple (i.e. in memory) storage context
    storage_context = StorageContext.from_defaults()

    # construct index
    index = VectorStoreIndex.from_documents(
        documents=documents, storage_context=storage_context
    )

    nodes = index.as_retriever().retrieve("test query str")

    # persist storage to disk
    storage_context.persist(str(tmp_path))

    # load storage context
    new_storage_context = StorageContext.from_defaults(persist_dir=str(tmp_path))

    # load index
    new_index = load_index_from_storage(new_storage_context)

    new_nodes = new_index.as_retriever().retrieve("test query str")

    assert nodes == new_nodes
