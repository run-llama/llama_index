from typing import List

import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
)
from llama_index.vector_stores.hnswlib import HnswlibVectorStore
import os


@pytest.fixture(scope="session")
def node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            embedding=[1.0, 0.0, 0.0],
            metadata={"external_id": "1"},
        ),
        TextNode(
            text="lorem ipsum",
            embedding=[0.0, 1.0, 0.0],
            metadata={"external_id": "12"},
        ),
        TextNode(
            text="lorem ipsum",
            embedding=[0.0, 0.5, 1.0],
            metadata={"external_id": "123"},
        ),
        TextNode(
            text="I was taught that the way of progress "
            + "was neither swift nor easy.",
            embedding=[0.0, 0.0, 0.9],
            metadata={"external_id": "4"},
        ),
        TextNode(
            text=(
                "The important thing is not to stop questioning."
                + " Curiosity has its own reason for existing."
            ),
            embedding=[0.0, 0.0, 0.5],
            metadata={"external_id": "5"},
        ),
        TextNode(
            text=(
                "I am no bird; and no net ensnares me;"
                + " I am a free human being with an independent will."
            ),
            embedding=[0.0, 0.0, 0.3],
            metadata={"external_id": "6"},
        ),
    ]


@pytest.fixture()
def hnswlib_store(node_embeddings: List[TextNode]) -> HnswlibVectorStore:
    space = "ip"
    dim = 3
    max_elements = len(node_embeddings) + 3
    vector_store = HnswlibVectorStore.from_params(space, dim, max_elements)
    vector_store.add(node_embeddings)
    return vector_store


def test_class():
    names_of_base_classes = [b.__name__ for b in HnswlibVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_add_nodes(hnswlib_store: HnswlibVectorStore):
    additional_node = TextNode(
        text=(
            "It's the possibility of having a dream come true "
            + "that makes life interesting."
        ),
        embedding=[0.7, 0.1, 0.0],
    )
    count_before = hnswlib_store._hnswlib_index.get_current_count()
    hnswlib_store.add([additional_node])
    assert count_before + 1 == hnswlib_store._hnswlib_index.get_current_count()


def test_query_nodes(hnswlib_store: HnswlibVectorStore):
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        similarity_top_k=hnswlib_store._hnswlib_index.get_current_count(),
    )
    result = hnswlib_store.query(query)
    assert result.similarities[0] == 0

    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        similarity_top_k=hnswlib_store._hnswlib_index.get_current_count(),
    )
    result = hnswlib_store.query(query, ef=100)
    assert result.similarities[0] == 0


def test_persistence(hnswlib_store: HnswlibVectorStore, tmp_path):
    persist_path = os.path.join(tmp_path, "storage", "hnswlib.index")
    assert not os.path.exists(persist_path)
    hnswlib_store.persist(persist_path)
    assert os.path.exists(persist_path)
    store_from_path = HnswlibVectorStore.from_persist_path(persist_path)
    assert set(hnswlib_store.client.get_ids_list()) == set(
        store_from_path.client.get_ids_list()
    )
    assert (
        hnswlib_store.client.get_max_elements()
        == store_from_path.client.get_max_elements()
    )
    assert (
        hnswlib_store.client.get_current_count()
        == store_from_path.client.get_current_count()
    )
