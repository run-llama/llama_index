"""Test MyScale indexes."""

from typing import List, cast

import pytest

try:
    import clickhouse_connect
except ImportError:
    clickhouse_connect = None  # type: ignore

from gpt_index.data_structs.data_structs_v2 import MyScaleIndexDict
from gpt_index.data_structs.node_v2 import Node
from gpt_index.indices.vector_store import GPTMyScaleIndex
from gpt_index.readers.schema.base import Document
from gpt_index.vector_stores import MyScaleVectorStore
from gpt_index.vector_stores.types import VectorStoreQuery

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
def indexDict() -> MyScaleIndexDict:
    return MyScaleIndexDict()


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
    index = cast(
        GPTMyScaleIndex,
        GPTMyScaleIndex.from_documents(documents, myscale_client=client),
    )
    response = index.query("What is?")
    assert str(response).strip() == ("What is what?")

    with pytest.raises(NotImplementedError):
        for doc in documents:
            index.delete(doc_id=cast(str, doc.doc_id))

    cast(MyScaleVectorStore, index._vector_store).drop()


@pytest.mark.skipif(
    clickhouse_connect is None
    or MYSCALE_CLUSTER_URL is None
    or MYSCALE_USERNAME is None
    or MYSCALE_CLUSTER_PASSWORD is None,
    reason="myscale-client not configured",
)
def test_init_without_documents(
    indexDict: MyScaleIndexDict, documents: List[Document]
) -> None:
    client = clickhouse_connect.get_client(
        host=MYSCALE_CLUSTER_URL,
        port=8443,
        username=MYSCALE_USERNAME,
        password=MYSCALE_CLUSTER_PASSWORD,
    )
    index = cast(
        GPTMyScaleIndex,
        GPTMyScaleIndex.from_documents(documents, myscale_client=client),
    )
    for doc in documents:
        index.insert(document=doc)
    response = index.query("What is?")
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
    index = cast(
        GPTMyScaleIndex,
        GPTMyScaleIndex.from_documents(documents, myscale_client=client),
    )
    query.query_embedding = index.service_context.embed_model.get_query_embedding(
        cast(str, query.query_str)
    )
    responseNodes = cast(List[Node], index._vector_store.query(query).nodes)
    assert len(responseNodes) == 1
    assert responseNodes[0].doc_id == "1"
    cast(MyScaleVectorStore, index._vector_store).drop()
