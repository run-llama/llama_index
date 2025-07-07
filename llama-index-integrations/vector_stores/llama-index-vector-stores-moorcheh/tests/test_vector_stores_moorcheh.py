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
from moorcheh_vector_store import MoorchehVectorStore

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
