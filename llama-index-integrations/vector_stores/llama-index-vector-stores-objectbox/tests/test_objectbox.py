import os
import shutil
from typing import Sequence

import pytest
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.vector_stores import VectorStoreQuery

from llama_index.vector_stores.objectbox import ObjectBoxVectorStore


EMBEDDING_DIM = 3


@pytest.fixture()
def vectorstore():
    obx = ObjectBoxVectorStore(embedding_dimensions=EMBEDDING_DIM)
    db_default_path = "objectbox"
    assert os.path.exists(
        db_default_path
    ), f"Directory '{db_default_path}' does not exist."
    filepath = os.path.join(db_default_path, "data.mdb")
    assert os.path.isfile(
        filepath
    ), f"File '{db_default_path}' not found in '{db_default_path}'"
    return obx


@pytest.fixture()
def node_embeddings() -> Sequence[BaseNode]:
    return [
        TextNode(
            id_="e8671c2d-8ee3-4f95-9730-7832f0115560",
            text="test1",
            embedding=[1.2, 0.3, -0.9],
        ),
        TextNode(
            id_="d0db4ed6-da16-4769-bf19-d1c06267a5f6",
            text="test2",
            embedding=[0.1, 0.0, 0.0],
        ),
        TextNode(
            id_="8601b27c-376e-48dd-a252-e61e01f29069",
            text="test3",
            embedding=[-2.3, 1.2, -6.7],
        ),
    ]


def test_add(vectorstore: ObjectBoxVectorStore, node_embeddings: Sequence[BaseNode]):
    node_ids = vectorstore.add(node_embeddings)
    retrieved_nodes = vectorstore.get_nodes(node_ids)
    assert len(retrieved_nodes) == len(node_embeddings)


def test_query(vectorstore: ObjectBoxVectorStore, node_embeddings: Sequence[BaseNode]):
    vectorstore.add(node_embeddings)
    search_result = vectorstore.query(
        VectorStoreQuery(query_embedding=[0.15, 0.001, -0.01], similarity_top_k=1)
    )
    assert len(search_result.ids) == 1
    assert search_result.nodes[0].id_ == "d0db4ed6-da16-4769-bf19-d1c06267a5f6"


def test_get_nodes(
    vectorstore: ObjectBoxVectorStore, node_embeddings: Sequence[BaseNode]
):
    vectorstore.add(node_embeddings)
    retrieved_nodes = vectorstore.get_nodes(
        node_ids=["8601b27c-376e-48dd-a252-e61e01f29069"]
    )
    assert len(retrieved_nodes) == 1
    assert retrieved_nodes[0].id_ == "8601b27c-376e-48dd-a252-e61e01f29069"


def test_count(vectorstore: ObjectBoxVectorStore, node_embeddings: Sequence[BaseNode]):
    vectorstore.add(node_embeddings)
    assert vectorstore.count() == len(node_embeddings)


def test_delete_nodes(
    vectorstore: ObjectBoxVectorStore, node_embeddings: Sequence[BaseNode]
):
    node_ids = vectorstore.add(node_embeddings)
    node_ids_to_be_deleted = node_ids[0:2]
    vectorstore.delete_nodes(node_ids_to_be_deleted)
    assert vectorstore.count() == 1


def test_clear(vectorstore: ObjectBoxVectorStore, node_embeddings: Sequence[BaseNode]):
    node_ids = vectorstore.add(node_embeddings)
    vectorstore.clear()
    retrieved_nodes = vectorstore.get_nodes(node_ids)
    assert len(retrieved_nodes) == 0


def remove_test_dir(test_dir: str):
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


@pytest.fixture(autouse=True)
def auto_cleanup(vectorstore: ObjectBoxVectorStore):
    yield  # run the test function
    vectorstore.close()
    os.remove(
        "llama-index-integrations/vector_stores/llama-index-vector-stores-objectbox/llama_index/vector_stores/objectbox/objectbox-model.json"
    )
    remove_test_dir("objectbox")
