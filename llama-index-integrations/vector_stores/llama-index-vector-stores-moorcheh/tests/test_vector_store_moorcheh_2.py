# Test.py

import pytest
import time
import uuid

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

# Import your custom vector store class
from llama_index.vector_stores.moorcheh import MoorchehVectorStore

should_skip = not os.getenv("MOORCHEH_API_KEY")

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
