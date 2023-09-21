from pathlib import Path
from typing import List

import pytest

from llama_index.indices.list.base import SummaryIndex
from llama_index.indices.loading import (
    load_index_from_storage,
    load_indices_from_storage,
)
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.schema import BaseNode, Document
from llama_index.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.storage.index_store.simple_index_store import SimpleIndexStore
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore


def test_load_index_from_storage_simple(
    documents: List[Document], tmp_path: Path, mock_service_context: ServiceContext
) -> None:
    # construct simple (i.e. in memory) storage context
    storage_context = StorageContext.from_defaults()

    # construct index
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=mock_service_context,
    )

    # persist storage to disk
    storage_context.persist(str(tmp_path))

    # load storage context
    new_storage_context = StorageContext.from_defaults(persist_dir=str(tmp_path))

    # load index
    new_index = load_index_from_storage(
        storage_context=new_storage_context, service_context=mock_service_context
    )

    assert index.index_id == new_index.index_id


def test_load_index_from_storage_multiple(
    nodes: List[BaseNode],
    tmp_path: Path,
    mock_service_context: ServiceContext,
) -> None:
    # construct simple (i.e. in memory) storage context
    storage_context = StorageContext.from_defaults()

    # add nodes to docstore
    storage_context.docstore.add_documents(nodes)

    # construct multiple indices
    vector_index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        service_context=mock_service_context,
    )
    vector_id = vector_index.index_id

    summary_index = SummaryIndex(
        nodes=nodes,
        storage_context=storage_context,
        service_context=mock_service_context,
    )

    list_id = summary_index.index_id

    # persist storage to disk
    storage_context.persist(str(tmp_path))

    # load storage context
    new_storage_context = StorageContext.from_defaults(persist_dir=str(tmp_path))

    # load single index should fail since there are multiple indices in index store
    with pytest.raises(ValueError):
        load_index_from_storage(
            new_storage_context, service_context=mock_service_context
        )

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
    documents: List[Document],
    tmp_path: Path,
    mock_service_context: ServiceContext,
) -> None:
    # construct simple (i.e. in memory) storage context
    storage_context = StorageContext.from_defaults()

    # construct index
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=mock_service_context,
    )

    nodes = index.as_retriever().retrieve("test query str")

    # persist storage to disk
    storage_context.persist(str(tmp_path))

    # load storage context
    new_storage_context = StorageContext.from_defaults(persist_dir=str(tmp_path))

    # load index
    new_index = load_index_from_storage(
        new_storage_context, service_context=mock_service_context
    )

    new_nodes = new_index.as_retriever().retrieve("test query str")

    assert nodes == new_nodes


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
def test_load_index_from_storage_faiss_vector_store(
    documents: List[Document],
    tmp_path: Path,
    mock_service_context: ServiceContext,
) -> None:
    import faiss

    # construct custom storage context
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore(),
        index_store=SimpleIndexStore(),
        vector_store=FaissVectorStore(faiss_index=faiss.IndexFlatL2(5)),
    )

    # construct index
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=mock_service_context,
    )

    nodes = index.as_retriever().retrieve("test query str")

    # persist storage to disk
    storage_context.persist(persist_dir=str(tmp_path))

    # load storage context
    new_storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(str(tmp_path)),
        index_store=SimpleIndexStore.from_persist_dir(str(tmp_path)),
        vector_store=FaissVectorStore.from_persist_dir(str(tmp_path)),
    )

    # load index
    new_index = load_index_from_storage(
        new_storage_context, service_context=mock_service_context
    )

    new_nodes = new_index.as_retriever().retrieve("test query str")

    assert nodes == new_nodes


def test_load_index_query_engine_service_context(
    documents: List[Document],
    tmp_path: Path,
    mock_service_context: ServiceContext,
) -> None:
    # construct simple (i.e. in memory) storage context
    storage_context = StorageContext.from_defaults()

    # construct index
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=mock_service_context,
    )

    # persist storage to disk
    storage_context.persist(str(tmp_path))

    # load storage context
    new_storage_context = StorageContext.from_defaults(persist_dir=str(tmp_path))

    # load index
    new_index = load_index_from_storage(
        storage_context=new_storage_context, service_context=mock_service_context
    )

    query_engine = index.as_query_engine()
    new_query_engine = new_index.as_query_engine()

    # make types happy
    assert isinstance(query_engine, RetrieverQueryEngine)
    assert isinstance(new_query_engine, RetrieverQueryEngine)
    # Ensure that the loaded index will end up querying with the same service_context
    assert (
        new_query_engine._response_synthesizer.service_context == mock_service_context
    )
