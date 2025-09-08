import pytest
import chromadb

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
import uuid


def _build_simple_collection():
    client = chromadb.EphemeralClient()
    name = f"chroma_mmr_test_{uuid.uuid4().hex[:8]}"
    col = client.get_or_create_collection(name)

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1],
        [0.0, 0.0, 1.0],
    ]
    ids = [f"id_{i}" for i in range(len(embeddings))]
    documents = [f"doc_{i}" for i in range(len(embeddings))]
    metadatas = [{"label": f"m{i}"} for i in range(len(embeddings))]  # non-empty

    col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return col


def test_chroma_mmr_happy_path_returns_k_results():
    col = _build_simple_collection()
    vs = ChromaVectorStore.from_collection(col)

    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
    )

    # Should not raise and should return exactly top_k results
    res = vs.query(query, mmr_threshold=0.5)

    # Either nodes or ids must be populated
    if res.ids is not None:
        assert len(res.ids) == 2
    elif res.nodes is not None:
        assert len(res.nodes) == 2
    else:
        pytest.fail("VectorStoreQueryResult must contain ids or nodes")


def test_chroma_mmr_conflicting_prefetch_params_raises():
    col = _build_simple_collection()
    vs = ChromaVectorStore.from_collection(col)

    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
    )

    with pytest.raises(ValueError):
        vs.query(query, mmr_threshold=0.5, mmr_prefetch_k=16, mmr_prefetch_factor=2.0)
