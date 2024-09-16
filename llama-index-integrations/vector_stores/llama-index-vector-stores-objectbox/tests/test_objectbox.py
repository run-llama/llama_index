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
    text_nodes_with_embeddings = [
        TextNode(text="test1", embedding=[1.2, 0.3, -0.9]),
        TextNode(text="test2", embedding=[0.1, 0.0, 0.0]),
        TextNode(text="test3", embedding=[-2.3, 1.2, -6.7]),
    ]
    return text_nodes_with_embeddings


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
    assert search_result.nodes[0].id_ == "2"


def test_get_nodes(
    vectorstore: ObjectBoxVectorStore, node_embeddings: Sequence[BaseNode]
):
    vectorstore.add(node_embeddings)
    retrieved_nodes = vectorstore.get_nodes(node_ids=["3"])
    assert len(retrieved_nodes) == 1
    assert retrieved_nodes[0].id_ == "3"


def test_delete(vectorstore: ObjectBoxVectorStore, node_embeddings: Sequence[BaseNode]):
    vectorstore.add(node_embeddings)
    vectorstore.delete(ref_doc_id="2")
    assert len(vectorstore.get_nodes(node_ids=["2"])) == 0


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
    print("Cleaned")
    vectorstore.close()
    os.remove("llama_index/vector_stores/objectbox/objectbox-model.json")
    remove_test_dir("objectbox")
