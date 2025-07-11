# Test.py

import os
import pytest
import time
import uuid
from typing import List

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
    FilterCondition,
    FilterOperator,
)

# Import your custom vector store class
from llama_index.vector_stores.moorcheh import MoorchehVectorStore

MAX_WAIT_TIME = 60
EMBED_DIM = 1536
MOORCHEH_API_KEY = os.environ.get("MOORCHEH_API_KEY", None)
should_skip = not MOORCHEH_API_KEY


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


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
def test_basic_e2e(index_with_nodes: VectorStoreIndex):
    nodes = index_with_nodes.as_retriever().retrieve("Hello, world 1!")
    assert len(nodes) >= 1  # Adjust if exact match count varies


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
def test_retrieval_with_filters(index_with_nodes: VectorStoreIndex):
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
    nodes = index_with_nodes.as_retriever(filters=filters).retrieve("Hello, world 1!")
    assert len(nodes) == 2

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="some_key",
                value=1,
                operator=FilterOperator.GT,
            ),
        ],
    )
    nodes = index_with_nodes.as_retriever(filters=filters).retrieve("Hello, world 1!")
    assert len(nodes) == 1

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="some_key",
                value=[1, 2],
                operator=FilterOperator.IN,
            ),
        ],
    )
    nodes = index_with_nodes.as_retriever(filters=filters).retrieve("Hello, world 1!")
    assert len(nodes) == 2

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="some_key",
                value="3",
                operator=FilterOperator.EQ,
            ),
        ],
    )
    nodes = index_with_nodes.as_retriever(filters=filters).retrieve("Hello, world 1!")
    assert len(nodes) == 1


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
def test_empty_retrieval(vector_store):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=[],  # no documents
        storage_context=storage_context,
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )

    results = index.as_retriever().retrieve("Nonexistent")
    assert results == []


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
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


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
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


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
def test_negative_filter_ops(index_with_nodes: VectorStoreIndex):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="some_key",
                value=2,
                operator=FilterOperator.NE,  # Not equal
            ),
        ]
    )
    nodes = index_with_nodes.as_retriever(filters=filters).retrieve("Hello")
    texts = [n.text for n in nodes]
    assert "Hello, world 2!" not in texts

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="some_key",
                value=[2, "3"],
                operator=FilterOperator.NOT_IN,
            ),
        ]
    )
    nodes = index_with_nodes.as_retriever(filters=filters).retrieve("Hello")
    texts = [n.text for n in nodes]
    assert "Hello, world 2!" not in texts
    assert "Hello, world 3!" not in texts


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
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
    assert len(res) >= 10  # fuzzy matching tolerance


# Test class inheritance
def test_class():
    assert BasePydanticVectorStore.__name__ in [
        b.__name__ for b in MoorchehVectorStore.__mro__
    ]


# Fixtures
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
    return MoorchehVectorStore(
        api_key=MOORCHEH_API_KEY,
        namespace=f"test-ns-{uuid.uuid4().hex[:8]}",
        namespace_type="vector",
        vector_dimension=EMBED_DIM,
        batch_size=100,
    )


@pytest.fixture
def index_with_nodes(vector_store, nodes: List[TextNode]):
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )
    time.sleep(2)
    return index


# Core Tests
@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
def test_basic_e2e(index_with_nodes: VectorStoreIndex):
    results = index_with_nodes.as_retriever().retrieve("Hello, world 1!")
    assert len(results) >= 1


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
def test_retrieval_with_filters(index_with_nodes: VectorStoreIndex):
    f1 = MetadataFilters(
        filters=[
            MetadataFilter(key="some_key", value=1, operator=FilterOperator.EQ),
            MetadataFilter(key="some_key", value=2, operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.OR,
    )
    assert len(index_with_nodes.as_retriever(filters=f1).retrieve("Hello")) == 2

    f2 = MetadataFilters(
        filters=[
            MetadataFilter(key="some_key", value=1, operator=FilterOperator.GT),
        ]
    )
    assert len(index_with_nodes.as_retriever(filters=f2).retrieve("Hello")) == 1

    f3 = MetadataFilters(
        filters=[
            MetadataFilter(key="some_key", value=[1, 2], operator=FilterOperator.IN),
        ]
    )
    assert len(index_with_nodes.as_retriever(filters=f3).retrieve("Hello")) == 2

    f4 = MetadataFilters(
        filters=[
            MetadataFilter(key="some_key", value="3", operator=FilterOperator.EQ),
        ]
    )
    assert len(index_with_nodes.as_retriever(filters=f4).retrieve("Hello")) == 1


# Additional Tests
@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
def test_empty_retrieval(vector_store):
    index = VectorStoreIndex(
        nodes=[],
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )
    assert index.as_retriever().retrieve("Nothing") == []


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
def test_namespace_isolation(nodes):
    ns1, ns2 = f"ns1-{uuid.uuid4().hex[:6]}", f"ns2-{uuid.uuid4().hex[:6]}"
    store1 = MoorchehVectorStore(
        api_key=MOORCHEH_API_KEY,
        namespace=ns1,
        namespace_type="vector",
        vector_dimension=EMBED_DIM,
    )
    store2 = MoorchehVectorStore(
        api_key=MOORCHEH_API_KEY,
        namespace=ns2,
        namespace_type="vector",
        vector_dimension=EMBED_DIM,
    )

    VectorStoreIndex(
        nodes=nodes[:1],
        storage_context=StorageContext.from_defaults(vector_store=store1),
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )
    VectorStoreIndex(
        nodes=nodes[1:],
        storage_context=StorageContext.from_defaults(vector_store=store2),
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )
    time.sleep(2)

    r1 = store1.as_retriever().retrieve("Hello")
    r2 = store2.as_retriever().retrieve("Hello")
    assert all("1" in n.text for n in r1)
    assert all("2" in n.text or "3" in n.text for n in r2)


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
def test_duplicate_upsert_behavior(vector_store):
    node = TextNode(
        id_="fixed-id",
        text="Duplicate node",
        metadata={"key": "val"},
        embedding=[0.1] * EMBED_DIM,
    )
    VectorStoreIndex(
        [node],
        StorageContext.from_defaults(vector_store=vector_store),
        MockEmbedding(EMBED_DIM),
    )
    VectorStoreIndex(
        [node],
        StorageContext.from_defaults(vector_store=vector_store),
        MockEmbedding(EMBED_DIM),
    )
    time.sleep(2)
    assert len(vector_store.as_retriever().retrieve("Duplicate")) >= 1


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
def test_conflicting_filters(index_with_nodes):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="some_key", value=1, operator=FilterOperator.EQ),
            MetadataFilter(key="some_key", value=2, operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.AND,
    )
    assert len(index_with_nodes.as_retriever(filters=filters).retrieve("Hello")) == 0


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
def test_similarity_vs_exact(index_with_nodes):
    results = index_with_nodes.as_retriever().retrieve("world")
    assert len(results) > 0


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
def test_filter_missing_metadata_key(index_with_nodes):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="nonexistent", value="missing", operator=FilterOperator.EQ
            )
        ]
    )
    assert len(index_with_nodes.as_retriever(filters=filters).retrieve("Hello")) == 0


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
def test_large_metadata_dict(vector_store):
    node = TextNode(
        text="Lots of metadata",
        metadata={f"key{i}": f"value{i}" for i in range(100)},
        embedding=[0.4] * EMBED_DIM,
    )
    VectorStoreIndex(
        [node],
        StorageContext.from_defaults(vector_store=vector_store),
        MockEmbedding(EMBED_DIM),
    )
    time.sleep(2)
    assert len(vector_store.as_retriever().retrieve("metadata")) >= 1


@pytest.mark.skipif(should_skip, reason="MOORCHEH_API_KEY not set")
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
    VectorStoreIndex(
        nodes,
        StorageContext.from_defaults(vector_store=store),
        MockEmbedding(EMBED_DIM),
    )
    time.sleep(5)
    assert len(store.as_retriever().retrieve("Node")) >= 10
