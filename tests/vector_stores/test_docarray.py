from typing import List, cast

import pytest

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.vector_stores import DocArrayInMemoryVectorStore, DocArrayHnswVectorStore
from llama_index.vector_stores.docarray.base import DocArrayVectorStore
from llama_index.vector_stores.types import NodeWithEmbedding, VectorStoreQuery


docarray = pytest.importorskip("docarray")

@pytest.fixture
def node_embeddings() -> List[NodeWithEmbedding]:
    return [
        NodeWithEmbedding(
            embedding=[1.0, 0.0, 0.0],
            node=Node(
                text="lorem ipsum",
                doc_id="c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
                relationships={DocumentRelationship.SOURCE: "test-0"},
            ),
        ),
        NodeWithEmbedding(
            embedding=[0.0, 1.0, 0.0],
            node=Node(
                text="lorem ipsum",
                doc_id="c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
                relationships={DocumentRelationship.SOURCE: "test-1"},
            ),
        ),
        NodeWithEmbedding(
            embedding=[0.0, 0.0, 1.0],
            node=Node(
                text="lorem ipsum",
                doc_id="c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
                relationships={DocumentRelationship.SOURCE: "test-2"},
            ),
        ),
    ]

def test_hnsw(node_embeddings: List[NodeWithEmbedding], tmp_path) -> None:
    docarray_vector_store = DocArrayHnswVectorStore(work_dir=tmp_path, dim=3)
    docarray_vector_store.add(node_embeddings)
    assert docarray_vector_store.num_docs() == 3

    query_emb = VectorStoreQuery(query_embedding=[0.0, 0.1, 0.0])
    res = docarray_vector_store.query(query_emb)

    assert len(res.nodes) == 1
    assert res.nodes[0].relationships[DocumentRelationship.SOURCE] == "test-1"

    docarray_vector_store.delete(doc_id="test-1")
    assert docarray_vector_store.num_docs() == 2

    new_vector_store = DocArrayHnswVectorStore(work_dir=tmp_path, dim=3)
    assert new_vector_store.num_docs() == 2

    new_vector_store.delete(doc_id="test-0")
    assert new_vector_store.num_docs() == 1


def test_in_memory(node_embeddings: List[NodeWithEmbedding], tmp_path) -> None:
    docarray_vector_store = DocArrayInMemoryVectorStore()
    docarray_vector_store.add(node_embeddings)
    assert docarray_vector_store.num_docs() == 3

    query_emb = VectorStoreQuery(query_embedding=[0.0, 0.1, 0.0])
    res = docarray_vector_store.query(query_emb)

    assert len(res.nodes) == 1
    assert res.nodes[0].relationships[DocumentRelationship.SOURCE] == "test-1"

    docarray_vector_store.delete(doc_id="test-1")
    assert docarray_vector_store.num_docs() == 2

    docarray_vector_store.persist(tmp_path/'index.bin')

    new_vector_store = DocArrayInMemoryVectorStore(index_path=tmp_path/'index.bin')
    assert new_vector_store.num_docs() == 2

    new_vector_store.delete(doc_id="test-0")
    assert new_vector_store.num_docs() == 1
