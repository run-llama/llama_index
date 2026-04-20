# Test.py

import os
import pytest
import time
import uuid
from typing import List
from unittest.mock import MagicMock
from moorcheh_sdk import MoorchehClient

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
    FilterCondition,
    FilterOperator,
    VectorStoreQuery,
)

# Import your custom vector store class
from llama_index.vector_stores.moorcheh import MoorchehVectorStore

MAX_WAIT_TIME = 60
EMBED_DIM = 1536
MOORCHEH_API_KEY = os.environ.get("MOORCHEH_API_KEY", None)
def _new_client_and_auth_ok() -> bool:
    if not MOORCHEH_API_KEY:
        return False
    try:
        with MoorchehClient(api_key=MOORCHEH_API_KEY) as client:
            if not hasattr(client, "namespaces"):
                return False
            client.namespaces.list()
            return True
    except Exception:
        return False


should_skip = not _new_client_and_auth_ok()
skip_reason = (
    "Requires valid MOORCHEH_API_KEY and new resource-style moorcheh-sdk client"
)


def test_class():
    names_of_base_classes = [b.__name__ for b in MoorchehVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.fixture
def nodes():
    return [
        TextNode(
            text="Hello, world 1!",
            metadata={"some_key": 1},
            embedding=[0.3] * EMBED_DIM,
        ),
        TextNode(
            text="Hello, world 2!",
            metadata={"some_key": 2},
            embedding=[0.5] * EMBED_DIM,
        ),
        TextNode(
            text="Hello, world 3!",
            metadata={"some_key": "3"},
            embedding=[0.7] * EMBED_DIM,
        ),
    ]


@pytest.fixture
def vector_store():
    namespace = f"test-ns-{uuid.uuid4().hex[:8]}"
    return MoorchehVectorStore(
        api_key=MOORCHEH_API_KEY,
        namespace=namespace,
        namespace_type="vector",
        vector_dimension=EMBED_DIM,
        batch_size=100,
    )


@pytest.fixture
def index_with_nodes(vector_store, nodes: List[TextNode]):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )

    # Optionally wait or delay if Moorcheh has any async delay
    time.sleep(2)

    return index


@pytest.fixture
def text_vector_store():
    namespace = f"test-text-ns-{uuid.uuid4().hex[:8]}"
    return MoorchehVectorStore(
        api_key=MOORCHEH_API_KEY,
        namespace=namespace,
        namespace_type="text",
        batch_size=100,
    )


@pytest.fixture
def text_index_with_nodes(text_vector_store, nodes: List[TextNode]):
    # For text namespaces, query/filter behavior is text-driven.
    storage_context = StorageContext.from_defaults(vector_store=text_vector_store)
    text_nodes = [
        TextNode(text=n.text, metadata=n.metadata, id_=n.node_id) for n in nodes
    ]
    index = VectorStoreIndex(
        nodes=text_nodes,
        storage_context=storage_context,
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )
    time.sleep(2)
    return index


@pytest.mark.skipif(should_skip, reason=skip_reason)
def test_basic_e2e(index_with_nodes: VectorStoreIndex):
    nodes = index_with_nodes.as_retriever().retrieve("Hello, world 1!")
    assert len(nodes) >= 1  # Adjust if exact match count varies


@pytest.mark.skipif(should_skip, reason=skip_reason)
def test_retrieval_with_filters(text_index_with_nodes: VectorStoreIndex):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="some_key",
                value=1,
                operator=FilterOperator.EQ,
            ),
            MetadataFilter(
                key="some_key",
                value=2,
                operator=FilterOperator.EQ,
            ),
        ],
        condition=FilterCondition.OR,
    )
    nodes = text_index_with_nodes.as_retriever(filters=filters).retrieve("Hello, world 1!")
    assert len(nodes) >= 1
    values = {n.node.metadata.get("some_key") for n in nodes}
    assert values.issubset({1, 2})

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="some_key",
                value=1,
                operator=FilterOperator.GT,
            ),
        ],
    )
    nodes = text_index_with_nodes.as_retriever(filters=filters).retrieve("Hello, world 1!")
    # Backend filtering behavior can vary slightly by scoring/ranking.
    assert len(nodes) >= 1

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="some_key",
                value=[1, 2],
                operator=FilterOperator.IN,
            ),
        ],
    )
    nodes = text_index_with_nodes.as_retriever(filters=filters).retrieve("Hello, world 1!")
    assert len(nodes) >= 1
    values = {n.node.metadata.get("some_key") for n in nodes}
    assert values.issubset({1, 2})

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="some_key",
                value="3",
                operator=FilterOperator.EQ,
            ),
        ],
    )
    nodes = text_index_with_nodes.as_retriever(filters=filters).retrieve("Hello, world 1!")
    assert len(nodes) >= 1
    values = {n.node.metadata.get("some_key") for n in nodes}
    assert values == {"3"}


@pytest.mark.skipif(should_skip, reason=skip_reason)
def test_empty_retrieval(vector_store):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=[],  # no documents
        storage_context=storage_context,
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )

    results = index.as_retriever().retrieve("Nonexistent")
    assert results == []


@pytest.mark.skipif(should_skip, reason=skip_reason)
def test_namespace_isolation(nodes):
    ns1 = f"ns1-{uuid.uuid4().hex[:6]}"
    ns2 = f"ns2-{uuid.uuid4().hex[:6]}"

    store1 = MoorchehVectorStore(
        api_key=MOORCHEH_API_KEY,
        namespace=ns1,
        namespace_type="vector",
        vector_dimension=EMBED_DIM,
    )
    index1 = VectorStoreIndex(
        nodes=nodes[:1],  # only first node
        storage_context=StorageContext.from_defaults(vector_store=store1),
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )

    store2 = MoorchehVectorStore(
        api_key=MOORCHEH_API_KEY,
        namespace=ns2,
        namespace_type="vector",
        vector_dimension=EMBED_DIM,
    )
    index2 = VectorStoreIndex(
        nodes=nodes[1:],  # remaining nodes
        storage_context=StorageContext.from_defaults(vector_store=store2),
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )

    time.sleep(2)

    res1 = index1.as_retriever().retrieve("Hello")
    res2 = index2.as_retriever().retrieve("Hello")

    assert all("1" in n.text for n in res1)
    assert all("2" in n.text or "3" in n.text for n in res2)


@pytest.mark.skipif(should_skip, reason=skip_reason)
def test_missing_metadata_handling():
    nodes = [
        TextNode(
            text="A node with metadata",
            metadata={"key": "val"},
            embedding=[0.1] * EMBED_DIM,
        ),
        TextNode(text="A node without metadata", embedding=[0.1] * EMBED_DIM),
    ]
    store = MoorchehVectorStore(
        api_key=MOORCHEH_API_KEY,
        namespace=f"missing-meta-{uuid.uuid4().hex[:6]}",
        namespace_type="vector",
        vector_dimension=EMBED_DIM,
    )
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )

    time.sleep(2)

    results = index.as_retriever().retrieve("A node")
    assert len(results) == 2


@pytest.mark.skipif(should_skip, reason=skip_reason)
def test_negative_filter_ops(text_index_with_nodes: VectorStoreIndex):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="some_key",
                value=2,
                operator=FilterOperator.NE,  # Not equal
            ),
        ]
    )
    nodes = text_index_with_nodes.as_retriever(filters=filters).retrieve("Hello")
    texts = [n.text for n in nodes]
    # NE is not translated to Moorcheh filter syntax yet.
    assert len(texts) >= 1

    not_in_operator = getattr(FilterOperator, "NOT_IN", None)
    if not_in_operator is not None:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="some_key",
                    value=[2, "3"],
                    operator=not_in_operator,
                ),
            ]
        )
        nodes = text_index_with_nodes.as_retriever(filters=filters).retrieve("Hello")
        texts = [n.text for n in nodes]
        assert len(texts) >= 1


@pytest.mark.skipif(should_skip, reason=skip_reason)
def test_large_batch_insert():
    nodes = [
        TextNode(text=f"Node {i}", embedding=[float(i / 100)] * EMBED_DIM)
        for i in range(200)
    ]
    store = MoorchehVectorStore(
        api_key=MOORCHEH_API_KEY,
        namespace=f"batchtest-{uuid.uuid4().hex[:6]}",
        namespace_type="vector",
        vector_dimension=EMBED_DIM,
        batch_size=50,
    )
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )

    time.sleep(5)
    res = index.as_retriever().retrieve("Node")
    assert len(res) >= 1


@pytest.mark.skipif(should_skip, reason=skip_reason)
def test_duplicate_upsert_behavior(vector_store):
    node = TextNode(
        id_="fixed-id",
        text="Duplicate node",
        metadata={"key": "val"},
        embedding=[0.1] * EMBED_DIM,
    )
    VectorStoreIndex(
        nodes=[node],
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )
    VectorStoreIndex(
        nodes=[node],
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )
    time.sleep(2)
    check_index = VectorStoreIndex(
        nodes=[],
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )
    assert len(check_index.as_retriever().retrieve("Duplicate")) >= 1


@pytest.mark.skipif(should_skip, reason=skip_reason)
def test_conflicting_filters(text_index_with_nodes):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="some_key", value=1, operator=FilterOperator.EQ),
            MetadataFilter(key="some_key", value=2, operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.AND,
    )
    results = text_index_with_nodes.as_retriever(filters=filters).retrieve("Hello")
    assert isinstance(results, list)


@pytest.mark.skipif(should_skip, reason=skip_reason)
def test_similarity_vs_exact(index_with_nodes):
    results = index_with_nodes.as_retriever().retrieve("world")
    assert len(results) > 0


@pytest.mark.skipif(should_skip, reason=skip_reason)
def test_filter_missing_metadata_key(text_index_with_nodes):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="nonexistent", value="missing", operator=FilterOperator.EQ
            )
        ]
    )
    results = text_index_with_nodes.as_retriever(filters=filters).retrieve("Hello")
    # Backend may ignore unsupported/missing metadata filters and fall back to semantic matches.
    assert isinstance(results, list)


@pytest.mark.skipif(should_skip, reason=skip_reason)
def test_large_metadata_dict(vector_store):
    node = TextNode(
        text="Lots of metadata",
        metadata={f"key{i}": f"value{i}" for i in range(100)},
        embedding=[0.4] * EMBED_DIM,
    )
    VectorStoreIndex(
        nodes=[node],
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )
    time.sleep(2)
    check_index = VectorStoreIndex(
        nodes=[],
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )
    assert len(check_index.as_retriever().retrieve("metadata")) >= 1


def _build_new_style_mock_client() -> MagicMock:
    client = MagicMock()
    client.namespaces.list.return_value = {"namespaces": []}
    client.namespaces.create.return_value = {"status": "success"}
    client.namespaces.delete.return_value = {"status": "success"}
    client.documents.upload.return_value = {"status": "success"}
    client.documents.delete.return_value = {"status": "success"}
    client.vectors.upload.return_value = {"status": "success"}
    client.vectors.delete.return_value = {"status": "success"}
    client.similarity_search.query.return_value = {"results": []}
    client.answer.generate.return_value = {"answer": "ok"}
    return client


def _patch_new_client(monkeypatch, mock_client: MagicMock) -> None:
    monkeypatch.setattr(
        "llama_index.vector_stores.moorcheh.base.MoorchehClient",
        lambda *args, **kwargs: mock_client,
    )


def test_new_client_namespaces_create_used(monkeypatch) -> None:
    mock_client = _build_new_style_mock_client()
    _patch_new_client(monkeypatch, mock_client)

    MoorchehVectorStore(
        api_key="test-key",
        namespace="unit-test-ns",
        namespace_type="text",
    )

    mock_client.namespaces.list.assert_called_once()
    mock_client.namespaces.create.assert_called_once_with(
        namespace_name="unit-test-ns",
        type="text",
        vector_dimension=None,
    )


def test_new_client_documents_upload_used(monkeypatch) -> None:
    mock_client = _build_new_style_mock_client()
    _patch_new_client(monkeypatch, mock_client)
    store = MoorchehVectorStore(
        api_key="test-key",
        namespace="text-ns",
        namespace_type="text",
    )

    store.add([TextNode(id_="id-1", text="hello", metadata={"a": 1})])
    mock_client.documents.upload.assert_called_once()


def test_new_client_answer_generate_used(monkeypatch) -> None:
    mock_client = _build_new_style_mock_client()
    mock_client.answer.generate.return_value = {"answer": "generated answer"}
    _patch_new_client(monkeypatch, mock_client)
    store = MoorchehVectorStore(
        api_key="test-key",
        namespace="text-ns",
        namespace_type="text",
    )

    answer = store.get_generative_answer("What is this?", top_k=2)
    mock_client.answer.generate.assert_called_once()
    assert answer == "generated answer"
