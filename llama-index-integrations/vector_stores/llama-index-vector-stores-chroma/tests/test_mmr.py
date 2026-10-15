import pytest
import chromadb

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
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


@pytest.mark.parametrize(
    "include",
    [[], ["documents", "metadatas", "distances"], ["embeddings", "uris"]],
)
@pytest.mark.parametrize("where", [None, {"label": "m1"}])
def test_chroma_mmr_accepts_include(include, where, mocker):
    col = _build_simple_collection()
    query_spy = mocker.spy(col, "query")
    vs = ChromaVectorStore.from_collection(col)
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
    )
    original_include = include.copy()

    result = vs.query(query, mmr_threshold=1.0, where=where, include=include)

    expected_ids = ["id_1"] if where else ["id_0", "id_1"]
    assert result.ids == expected_ids
    assert [node.text for node in result.nodes] == [
        f"doc_{node_id.removeprefix('id_')}" for node_id in expected_ids
    ]
    assert len(result.similarities) == len(expected_ids)
    assert set(query_spy.call_args.kwargs["include"]) == set(include) | {
        "embeddings",
        "documents",
        "metadatas",
        "distances",
    }
    assert include == original_include


def test_chroma_mmr_retriever_accepts_include():
    store = ChromaVectorStore.from_collection(_build_simple_collection())
    embed_model = MockEmbedding(embed_dim=3)
    index = VectorStoreIndex.from_vector_store(
        store, embed_model=embed_model, transformations=[embed_model]
    )
    retriever_kwargs = {
        "similarity_top_k": 2,
        "vector_store_query_mode": VectorStoreQueryMode.MMR,
    }
    expected = index.as_retriever(**retriever_kwargs).retrieve("example query")
    result = index.as_retriever(
        **retriever_kwargs,
        vector_store_kwargs={"include": ["documents", "metadatas", "distances"]},
    ).retrieve("example query")

    assert len(result) == 2
    assert [node.node_id for node in result] == [node.node_id for node in expected]
    assert [node.score for node in result] == pytest.approx(
        [node.score for node in expected]
    )
