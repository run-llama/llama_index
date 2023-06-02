"""Test MyScale indexes."""

from typing import List, cast

import pytest

from llama_index.indices.vector_store.base import GPTVectorStoreIndex
from llama_index.storage.storage_context import StorageContext

try:
    import clickhouse_connect
except ImportError:
    clickhouse_connect = None  # type: ignore

from llama_index.data_structs.node import Node
from llama_index.readers.schema.base import Document
from llama_index.vector_stores import MyScaleVectorStore
from llama_index.vector_stores.types import VectorStoreQuery

# local test only, update variable here for test
MYSCALE_CLUSTER_URL = None
MYSCALE_USERNAME = None
MYSCALE_CLUSTER_PASSWORD = None


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(doc_id="1", text=doc_text)]


@pytest.fixture
def query() -> VectorStoreQuery:
    return VectorStoreQuery(query_str="What is?", doc_ids=["1"])


@pytest.mark.skipif(
    clickhouse_connect is None
    or MYSCALE_CLUSTER_URL is None
    or MYSCALE_USERNAME is None
    or MYSCALE_CLUSTER_PASSWORD is None,
    reason="myscale-client not configured",
)
def test_overall_workflow(documents: List[Document]) -> None:
    client = clickhouse_connect.get_client(
        host=MYSCALE_CLUSTER_URL,
        port=8443,
        username=MYSCALE_USERNAME,
        password=MYSCALE_CLUSTER_PASSWORD,
    )
    vector_store = MyScaleVectorStore(myscale_client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = GPTVectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    query_engine = index.as_query_engine()
    response = query_engine.query("What is?")
    assert str(response).strip() == ("What is what?")

    with pytest.raises(NotImplementedError):
        for doc in documents:
            index.delete_ref_doc(ref_doc_id=cast(str, doc.doc_id))

    cast(MyScaleVectorStore, index._vector_store).drop()


@pytest.mark.skipif(
    clickhouse_connect is None
    or MYSCALE_CLUSTER_URL is None
    or MYSCALE_USERNAME is None
    or MYSCALE_CLUSTER_PASSWORD is None,
    reason="myscale-client not configured",
)
def test_init_without_documents(documents: List[Document]) -> None:
    client = clickhouse_connect.get_client(
        host=MYSCALE_CLUSTER_URL,
        port=8443,
        username=MYSCALE_USERNAME,
        password=MYSCALE_CLUSTER_PASSWORD,
    )
    vector_store = MyScaleVectorStore(myscale_client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = GPTVectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    for doc in documents:
        index.insert(document=doc)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is?")
    assert str(response).strip() == ("What is what?")

    cast(MyScaleVectorStore, index._vector_store).drop()


@pytest.mark.skipif(
    clickhouse_connect is None
    or MYSCALE_CLUSTER_URL is None
    or MYSCALE_USERNAME is None
    or MYSCALE_CLUSTER_PASSWORD is None,
    reason="myscale-client not configured",
)
def test_myscale_combine_search(
    documents: List[Document], query: VectorStoreQuery
) -> None:
    client = clickhouse_connect.get_client(
        host=MYSCALE_CLUSTER_URL,
        port=8443,
        username=MYSCALE_USERNAME,
        password=MYSCALE_CLUSTER_PASSWORD,
    )
    vector_store = MyScaleVectorStore(myscale_client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = GPTVectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    query.query_embedding = index.service_context.embed_model.get_query_embedding(
        cast(str, query.query_str)
    )
    responseNodes = cast(List[Node], index._vector_store.query(query).nodes)
    assert len(responseNodes) == 1
    assert responseNodes[0].doc_id == "1"
    cast(MyScaleVectorStore, index._vector_store).drop()
