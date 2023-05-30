import os
from pathlib import Path
from typing import List

import pytest

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.vector_stores import (
    DocArrayHnswVectorStore,
    DocArrayInMemoryVectorStore,
)
from llama_index.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    NodeWithEmbedding,
    VectorStoreQuery,
)

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
                extra_info={
                    "author": "Stephen King",
                    "theme": "Friendship",
                },
            ),
        ),
        NodeWithEmbedding(
            embedding=[0.0, 1.0, 0.0],
            node=Node(
                text="lorem ipsum",
                doc_id="c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
                relationships={DocumentRelationship.SOURCE: "test-1"},
                extra_info={
                    "director": "Francis Ford Coppola",
                    "theme": "Mafia",
                },
            ),
        ),
        NodeWithEmbedding(
            embedding=[0.0, 0.0, 1.0],
            node=Node(
                text="lorem ipsum",
                doc_id="c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
                relationships={DocumentRelationship.SOURCE: "test-2"},
                extra_info={
                    "director": "Christopher Nolan",
                },
            ),
        ),
    ]


def test_hnsw(node_embeddings: List[NodeWithEmbedding], tmp_path: Path) -> None:
    docarray_vector_store = DocArrayHnswVectorStore(work_dir=str(tmp_path), dim=3)
    docarray_vector_store.add(node_embeddings)
    assert docarray_vector_store.num_docs() == 3

    query_emb = VectorStoreQuery(query_embedding=[0.0, 0.1, 0.0])
    res = docarray_vector_store.query(query_emb)

    assert len(res.nodes) == 1  # type: ignore[arg-type]
    rf = res.nodes[0].relationships[DocumentRelationship.SOURCE]  # type: ignore[index]
    assert rf == "test-1"

    docarray_vector_store.delete(doc_id="test-1")
    assert docarray_vector_store.num_docs() == 2

    new_vector_store = DocArrayHnswVectorStore(work_dir=str(tmp_path), dim=3)
    assert new_vector_store.num_docs() == 2

    new_vector_store.delete(doc_id="test-0")
    assert new_vector_store.num_docs() == 1


def test_in_memory(node_embeddings: List[NodeWithEmbedding], tmp_path: Path) -> None:
    docarray_vector_store = DocArrayInMemoryVectorStore()
    docarray_vector_store.add(node_embeddings)
    assert docarray_vector_store.num_docs() == 3

    query_emb = VectorStoreQuery(query_embedding=[0.0, 0.1, 0.0])
    res = docarray_vector_store.query(query_emb)

    assert len(res.nodes) == 1  # type: ignore[arg-type]
    rf = res.nodes[0].relationships[DocumentRelationship.SOURCE]  # type: ignore[index]
    assert rf == "test-1"

    docarray_vector_store.delete(doc_id="test-1")
    assert docarray_vector_store.num_docs() == 2

    docarray_vector_store.persist(os.path.join(str(tmp_path), "index.bin"))

    new_vector_store = DocArrayInMemoryVectorStore(
        index_path=os.path.join(str(tmp_path), "index.bin")
    )
    assert new_vector_store.num_docs() == 2

    new_vector_store.delete(doc_id="test-0")
    assert new_vector_store.num_docs() == 1


def test_in_memory_filters(node_embeddings: List[NodeWithEmbedding]) -> None:
    docarray_vector_store = DocArrayInMemoryVectorStore()
    docarray_vector_store.add(node_embeddings)
    assert docarray_vector_store.num_docs() == 3

    filters = MetadataFilters(filters=[ExactMatchFilter(key="theme", value="Mafia")])

    query_emb = VectorStoreQuery(query_embedding=[0.0, 0.1, 0.0], filters=filters)
    res = docarray_vector_store.query(query_emb)

    assert len(res.nodes) == 1  # type: ignore[arg-type]
    assert res.nodes[0].extra_info["theme"] == "Mafia"  # type: ignore[index]
    rf = res.nodes[0].relationships[DocumentRelationship.SOURCE]  # type: ignore[index]
    assert rf == "test-1"


def test_hnsw_filters(node_embeddings: List[NodeWithEmbedding], tmp_path: Path) -> None:
    docarray_vector_store = DocArrayHnswVectorStore(work_dir=str(tmp_path), dim=3)
    docarray_vector_store.add(node_embeddings)
    assert docarray_vector_store.num_docs() == 3

    filters = MetadataFilters(filters=[ExactMatchFilter(key="theme", value="Mafia")])

    query_emb = VectorStoreQuery(query_embedding=[0.0, 0.1, 0.0], filters=filters)
    res = docarray_vector_store.query(query_emb)

    assert len(res.nodes) == 1  # type: ignore[arg-type]
    assert res.nodes[0].extra_info["theme"] == "Mafia"  # type: ignore[index]
    rf = res.nodes[0].relationships[DocumentRelationship.SOURCE]  # type: ignore[index]
    assert rf == "test-1"
