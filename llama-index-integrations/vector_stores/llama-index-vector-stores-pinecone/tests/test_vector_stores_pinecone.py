import os
import pytest
import time
import uuid
import pinecone.db_data

from pinecone import Pinecone, ServerlessSpec
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
from llama_index.vector_stores.pinecone import PineconeVectorStore

MAX_WAIT_TIME = 60
EMBED_DIM = 1536
PINECONE_API_KEY = os.environ.get(
    "PINECONE_API_KEY",
    None,
)
should_skip = not all((PINECONE_API_KEY,))


def test_class():
    names_of_base_classes = [b.__name__ for b in PineconeVectorStore.__mro__]
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
def pinecone_index():
    index_name = f"{uuid.uuid4()}"

    pc = Pinecone(api_key=PINECONE_API_KEY)
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    pc_index = pc.Index(index_name)

    yield pc_index

    pc.delete_index(index_name)


@pytest.fixture
def index_with_nodes(
    pinecone_index: pinecone.db_data.index.Index, nodes: List[TextNode]
):
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )

    # Note: not ideal, but pinecone takes a while to index the nodes
    start_time = time.time()
    while True:
        stats = pinecone_index.describe_index_stats()
        if stats["total_vector_count"] != len(nodes):
            if time.time() - start_time > MAX_WAIT_TIME:
                raise Exception("Index not ready after 60 seconds")

            time.sleep(1)
        else:
            break

    return index


@pytest.mark.skipif(
    should_skip, reason="PINECONE_API_KEY and/or PINECONE_INDEX_NAME not set"
)
def test_basic_e2e(index_with_nodes: VectorStoreIndex):
    nodes = index_with_nodes.as_retriever().retrieve("Hello, world 1!")
    assert len(nodes) == 2


@pytest.mark.skipif(
    should_skip, reason="PINECONE_API_KEY and/or PINECONE_INDEX_NAME not set"
)
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
