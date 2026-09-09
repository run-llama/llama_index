"""Tests for :class:`InfinoVectorStore`, exercised against the embedded engine.

All retrieval tests run in local embedded mode over a temp directory with
deterministic near-orthogonal vectors — no embedding model, no network. The
constructor-mode test monkeypatches ``infino.connect`` to verify the local /
bucket / hosted branches build the right call without live infrastructure.
"""

import tempfile

import infino
import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from llama_index.vector_stores.infino import InfinoVectorStore

DIM = 16


def _unit_vector(i: int) -> list:
    """A near-orthogonal one-hot vector in DIM dimensions."""
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return v


def _make_nodes():
    specs = [
        ("alpha the quick brown fox", "animals", "src-a"),
        ("beta electric vehicles charge", "tech", "src-b"),
        ("gamma the lazy dog sleeps", "animals", "src-c"),
    ]
    nodes = []
    for i, (text, category, ref) in enumerate(specs):
        node = TextNode(
            id_=f"node-{i}",
            text=text,
            metadata={"category": category, "rank": i},
            embedding=_unit_vector(i),
        )
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=ref)
        nodes.append(node)
    return nodes


@pytest.fixture
def store():
    tmp = tempfile.mkdtemp()
    s = InfinoVectorStore(
        tmp,
        table_name="t_test",
        embed_dim=DIM,
        metric="cosine",
        metadata_columns=["category"],
    )
    s.add(_make_nodes())
    return s


def test_default_vector_query_returns_nearest_with_text(store):
    # Query nearest to node-0 (one-hot at index 0).
    q = VectorStoreQuery(query_embedding=_unit_vector(0), similarity_top_k=2)
    res = store.query(q)
    assert res.ids[0] == "node-0"
    assert len(res.similarities) == len(res.nodes) == 2
    # stores_text: the node text comes back with no separate docstore.
    assert res.nodes[0].get_content() == "alpha the quick brown fox"
    # cosine self-match ~= similarity 1.0
    assert res.similarities[0] > 0.99
    # metadata round-trips (promoted + JSON).
    assert res.nodes[0].metadata["category"] == "animals"
    assert res.nodes[0].metadata["rank"] == 0


def test_hybrid_query_returns_results(store):
    q = VectorStoreQuery(
        query_str="electric vehicles",
        query_embedding=_unit_vector(1),
        similarity_top_k=3,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = store.query(q)
    assert len(res.nodes) > 0
    assert "node-1" in res.ids
    # text present on hybrid results too
    assert all(n.get_content() for n in res.nodes)


def test_text_search_bm25(store):
    q = VectorStoreQuery(
        query_str="lazy dog",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.TEXT_SEARCH,
    )
    res = store.query(q)
    assert "node-2" in res.ids


def test_metadata_filter_on_promoted_column(store):
    flt = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vector(0), similarity_top_k=5, filters=flt
    )
    res = store.query(q)
    assert len(res.nodes) == 1
    assert res.ids == ["node-1"]
    assert res.nodes[0].metadata["category"] == "tech"


def test_filter_on_undeclared_column_raises(store):
    flt = MetadataFilters(
        filters=[MetadataFilter(key="rank", value=0, operator=FilterOperator.EQ)]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vector(0), similarity_top_k=5, filters=flt
    )
    with pytest.raises(ValueError):
        store.query(q)


def test_filter_with_hybrid_mode_raises(store):
    # Filters route through the vector path only; a filter on hybrid/text must
    # raise rather than be silently dropped.
    flt = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)
        ]
    )
    q = VectorStoreQuery(
        query_str="electric vehicles",
        query_embedding=_unit_vector(1),
        similarity_top_k=3,
        filters=flt,
        mode=VectorStoreQueryMode.HYBRID,
    )
    with pytest.raises(NotImplementedError):
        store.query(q)


def test_delete_by_ref_doc_id(store):
    store.delete("src-a")
    q = VectorStoreQuery(query_embedding=_unit_vector(0), similarity_top_k=5)
    res = store.query(q)
    assert "node-0" not in res.ids
    # the other nodes survive
    assert "node-1" in res.ids and "node-2" in res.ids


def test_delete_nodes_by_id(store):
    store.delete_nodes(node_ids=["node-1"])
    q = VectorStoreQuery(query_embedding=_unit_vector(1), similarity_top_k=5)
    res = store.query(q)
    assert "node-1" not in res.ids


def test_index_round_trip_no_docstore(store):
    # Prove VectorStoreIndex.from_vector_store returns node text with no docstore.
    from llama_index.core import VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding

    index = VectorStoreIndex.from_vector_store(
        store, embed_model=MockEmbedding(embed_dim=DIM)
    )
    retriever = index.as_retriever(similarity_top_k=3)
    # MockEmbedding maps any query to a constant vector; we only assert that
    # retrieved nodes carry their text back from the single Infino table.
    results = retriever.retrieve("anything")
    assert len(results) > 0
    assert all(r.node.get_content() for r in results)


def test_async_add_and_query(store):
    import asyncio

    async def run():
        q = VectorStoreQuery(query_embedding=_unit_vector(2), similarity_top_k=1)
        return await store.aquery(q)

    res = asyncio.run(run())
    assert res.ids[0] == "node-2"


def test_stores_text_flag():
    tmp = tempfile.mkdtemp()
    s = InfinoVectorStore(tmp, table_name="flag")
    assert s.stores_text is True


def test_persistence_reopen_infers_metadata_columns():
    # A fresh store on the same dir + table (an app restart) must work WITHOUT
    # re-declaring metadata_columns: promoted metadata round-trips and stays
    # filterable because the schema is read back from the existing table.
    tmp = tempfile.mkdtemp()
    s1 = InfinoVectorStore(
        tmp,
        table_name="persist",
        embed_dim=DIM,
        metric="cosine",
        metadata_columns=["category"],
    )
    s1.add(_make_nodes())

    s2 = InfinoVectorStore(tmp, table_name="persist", metric="cosine")
    res = s2.query(
        VectorStoreQuery(query_embedding=_unit_vector(0), similarity_top_k=2)
    )
    assert res.ids[0] == "node-0"
    assert res.nodes[0].get_content() == "alpha the quick brown fox"
    # promoted metadata survives the reopen without being re-declared
    assert res.nodes[0].metadata["category"] == "animals"

    flt = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)
        ]
    )
    res2 = s2.query(
        VectorStoreQuery(
            query_embedding=_unit_vector(0), similarity_top_k=5, filters=flt
        )
    )
    assert res2.ids == ["node-1"]


def test_end_to_end_from_documents():
    # The canonical LlamaIndex path: from_documents drives chunk -> embed -> add
    # through a StorageContext, then retrieval returns nodes with their text.
    from llama_index.core import Document, StorageContext, VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding

    tmp = tempfile.mkdtemp()
    store = InfinoVectorStore(tmp, table_name="e2e", embed_dim=DIM)
    storage_context = StorageContext.from_defaults(vector_store=store)
    docs = [
        Document(text="the quick brown fox jumps over the lazy dog"),
        Document(text="electric vehicles charge overnight at home"),
    ]
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=MockEmbedding(embed_dim=DIM),
    )
    nodes = index.as_retriever(similarity_top_k=2).retrieve("anything")
    assert len(nodes) > 0
    assert all(n.node.get_content() for n in nodes)


def test_constructor_connect_modes(monkeypatch):
    """Local / bucket / hosted URIs build the right infino.connect call."""
    calls = []

    class _FakeConn:
        pass

    def fake_connect(uri, **kwargs):
        calls.append((uri, kwargs))
        return _FakeConn()

    monkeypatch.setattr(infino, "connect", fake_connect)

    # Local embedded — plain path, no api_key / storage_options.
    InfinoVectorStore("/tmp/some/dir", table_name="x")
    uri, kwargs = calls[-1]
    assert uri == "/tmp/some/dir"
    assert kwargs == {}

    # Object-storage embedded — storage_options passed through.
    opts = {"aws_region": "us-east-1"}
    InfinoVectorStore("s3://bucket/prefix", table_name="x", storage_options=opts)
    uri, kwargs = calls[-1]
    assert uri == "s3://bucket/prefix"
    assert kwargs == {"storage_options": opts}

    # Hosted cloud — api_key passed through.
    InfinoVectorStore("https://api.platform.infino.ws/ws", table_name="x", api_key="secret")
    uri, kwargs = calls[-1]
    assert uri == "https://api.platform.infino.ws/ws"
    assert kwargs == {"api_key": "secret"}


def test_constructor_api_key_file(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(infino, "connect", lambda uri, **kw: calls.append((uri, kw)))
    key_file = tmp_path / "key.txt"
    key_file.write_text("file-secret\n")
    InfinoVectorStore("https://api.platform.infino.ws/ws", api_key_file=str(key_file))
    _, kwargs = calls[-1]
    assert kwargs == {"api_key": "file-secret"}
