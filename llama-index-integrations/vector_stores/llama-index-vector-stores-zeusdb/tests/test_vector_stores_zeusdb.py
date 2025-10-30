# tests/test_vector_stores_zeusdb.py
from dataclasses import dataclass
import importlib
import math
import sys
import types

import pytest


# -----------------------------
# Minimal in-memory ZeusDB fake
# -----------------------------
def _cosine_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    sim = dot / (na * nb)
    return 1.0 - sim  # adapter expects distance; later converts back to similarity


class _FakeIndex:
    def __init__(self, dim, space="cosine", **kwargs):
        self.dim = dim
        self.space = space
        self.records = []  # list of dict: {id, vector, metadata}
        self._next_id = 1
        self._saved_path = None

    def add(self, payload, overwrite=True):
        vectors = payload.get("vectors", [])
        metadatas = payload.get("metadatas", [])
        ids = payload.get("ids", None)

        assigned = []
        for i, v in enumerate(vectors):
            meta = dict(metadatas[i]) if i < len(metadatas) else {}
            if ids and ids[i]:
                rid = str(ids[i])
            else:
                rid = str(self._next_id)
                self._next_id += 1
            # overwrite means replace if existing id matches
            if overwrite:
                self.records = [r for r in self.records if r["id"] != rid]
            self.records.append({"id": rid, "vector": list(v), "metadata": meta})
            assigned.append(rid)
        return {"ids": assigned}

    def remove_point(self, node_id):
        """Remove a point by ID - ADDED FOR DELETE SUPPORT"""
        initial_count = len(self.records)
        self.records = [r for r in self.records if r["id"] != str(node_id)]
        removed = initial_count > len(self.records)
        return removed  # Return True if something was removed

    def _match_filter(self, meta, zfilter):
        """Match filter in FLAT format (like actual Rust code)."""
        if not zfilter:
            return True

        # Iterate over flat filter dict
        for field, condition in zfilter.items():
            field_value = meta.get(field)

            # Direct value = equality check
            if isinstance(condition, str | int | float | bool | type(None)):
                if field_value != condition:
                    return False
            # Operator dict
            elif isinstance(condition, dict):
                for op, target_value in condition.items():
                    if op == "eq" and field_value != target_value:
                        return False
                    elif op == "ne" and field_value == target_value:
                        return False
                    elif op == "gt" and not (
                        isinstance(field_value, int | float)
                        and field_value > target_value
                    ):
                        return False
                    elif op == "gte" and not (
                        isinstance(field_value, int | float)
                        and field_value >= target_value
                    ):
                        return False
                    elif op == "lt" and not (
                        isinstance(field_value, int | float)
                        and field_value < target_value
                    ):
                        return False
                    elif op == "lte" and not (
                        isinstance(field_value, int | float)
                        and field_value <= target_value
                    ):
                        return False
                    elif op == "in" and field_value not in target_value:
                        return False
                    elif op == "contains":
                        if (
                            isinstance(field_value, list)
                            and target_value not in field_value
                        ):
                            return False
                        elif (
                            isinstance(field_value, str)
                            and str(target_value) not in field_value
                        ):
                            return False
                        else:
                            return False

        return True

    def search(self, vector, top_k=5, filter=None, ef_search=None, return_vector=False):
        # filter records
        recs = [r for r in self.records if self._match_filter(r["metadata"], filter)]
        # score
        scored = []
        for r in recs:
            if self.space == "cosine":
                dist = _cosine_distance(vector, r["vector"])
            else:
                # simple L2
                dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(vector, r["vector"])))
            res = {"id": r["id"], "distance": dist}
            if return_vector:
                res["vector"] = list(r["vector"])
            scored.append(res)
        scored.sort(key=lambda x: x["distance"])
        return {"results": scored[:top_k]}

    def get_records(self, ids, return_vector=False):
        """Get records by IDs - needed for node reconstruction."""
        if isinstance(ids, str):
            ids = [ids]

        results = []
        id_set = set(str(i) for i in ids)

        for record in self.records:
            if str(record["id"]) in id_set:
                result = {
                    "id": record["id"],
                    "metadata": record["metadata"].copy(),
                    "score": 0.0,
                }
                if return_vector:
                    result["vector"] = record["vector"]
                results.append(result)

        return results

    def clear(self):
        self.records.clear()

    def save(self, path):
        self._saved_path = path  # track that save was called

    def get_vector_count(self):
        return len(self.records)

    def get_stats(self):
        return {"dim": self.dim, "space": self.space, "count": len(self.records)}


class _FakeVectorDatabase:
    def create(self, index_type="hnsw", dim=None, space="cosine", **kwargs):
        if dim is None:
            raise ValueError("dim required")
        return _FakeIndex(dim=dim, space=space, **kwargs)

    def load(self, path):
        # In a "real" load we'd deserialize. For tests, hand back an empty index.
        return _FakeIndex(dim=8, space="cosine")


@pytest.fixture(autouse=True)
def fake_zeusdb_module(monkeypatch):
    """
    Install a fake `zeusdb` module *before* importing the adapter.
    """
    mod = types.ModuleType("zeusdb")
    mod.VectorDatabase = _FakeVectorDatabase  # type: ignore[attr-defined]
    # optional logging_config fallback (adapter handles absence)
    sys.modules["zeusdb"] = mod
    yield
    sys.modules.pop("zeusdb", None)


# -------------------------------------
# Import the adapter after faking ZeusDB
# -------------------------------------
@pytest.fixture
def ZeusDBVectorStore():
    # reload our package to bind to the fake module
    import llama_index.vector_stores.zeusdb.base as zeusdb_base

    importlib.reload(zeusdb_base)
    return zeusdb_base.ZeusDBVectorStore


# --------------------
# Minimal node helper
# --------------------
@dataclass
class FakeNode:
    text: str
    embedding: list[float]
    metadata: dict
    id_: str | None = None
    ref_doc_id: str | None = None


# ----------
# Test cases
# ----------
def test_add_and_query_basic(ZeusDBVectorStore):
    store = ZeusDBVectorStore(dim=3, distance="cosine")
    n1 = FakeNode("a", [1.0, 0.0, 0.0], {"category": "x"}, id_="1", ref_doc_id="docA")
    n2 = FakeNode("b", [0.9, 0.1, 0.0], {"category": "y"}, id_="2", ref_doc_id="docA")
    out_ids = store.add([n1, n2])
    assert set(out_ids) == {"1", "2"}

    q = [1.0, 0.0, 0.0]
    from llama_index.core.vector_stores.types import VectorStoreQuery

    res = store.query(VectorStoreQuery(query_embedding=q, similarity_top_k=1))
    assert res.ids and res.similarities
    assert res.ids[0] in {"1", "2"}
    # nearest to [1,0,0] should be id "1"
    assert res.ids[0] == "1"


def test_filters_and_delete_nodes(ZeusDBVectorStore):
    store = ZeusDBVectorStore(dim=3)
    nodes = [
        FakeNode("a", [1.0, 0.0, 0.0], {"category": "x"}, id_="1", ref_doc_id="docA"),
        FakeNode("b", [0.0, 1.0, 0.0], {"category": "y"}, id_="2", ref_doc_id="docB"),
        FakeNode("c", [0.0, 0.0, 1.0], {"category": "x"}, id_="3", ref_doc_id="docC"),
    ]
    store.add(nodes)

    from llama_index.core.vector_stores.types import (
        FilterCondition,
        FilterOperator,
        MetadataFilters,
        VectorStoreQuery,
    )

    mf = MetadataFilters.from_dicts(
        [
            {"key": "category", "value": "x", "operator": FilterOperator.EQ},
        ],
        condition=FilterCondition.AND,
    )
    res = store.query(
        VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0],
            similarity_top_k=10,
            filters=mf,
        )
    )
    assert set(res.ids) == {"1", "3"}  # filter applied

    # Now delete node id "3" via delete_nodes
    store.delete_nodes(node_ids=["3"])
    res2 = store.query(
        VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0],
            similarity_top_k=10,
            filters=mf,
        )
    )
    assert set(res2.ids) == {"1"}


def test_delete_by_ref_doc_id_not_supported(ZeusDBVectorStore):
    """Test that delete by ref_doc_id raises NotImplementedError."""
    store = ZeusDBVectorStore(dim=3)
    nodes = [
        FakeNode("a", [1.0, 0.0, 0.0], {"k": 1}, id_="1", ref_doc_id="docA"),
        FakeNode("b", [0.0, 1.0, 0.0], {"k": 2}, id_="2", ref_doc_id="docA"),
        FakeNode("c", [0.0, 0.0, 1.0], {"k": 3}, id_="3", ref_doc_id="docB"),
    ]
    store.add(nodes)

    # Should raise NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        store.delete("docA")

    assert "ref_doc_id" in str(exc_info.value).lower()
    assert "remove_point" in str(exc_info.value).lower()


def test_delete_nodes_by_id_works(ZeusDBVectorStore):
    """Test that delete_nodes by ID works correctly."""
    store = ZeusDBVectorStore(dim=3)
    nodes = [
        FakeNode("a", [1.0, 0.0, 0.0], {"k": 1}, id_="1", ref_doc_id="docA"),
        FakeNode("b", [0.0, 1.0, 0.0], {"k": 2}, id_="2", ref_doc_id="docA"),
        FakeNode("c", [0.0, 0.0, 1.0], {"k": 3}, id_="3", ref_doc_id="docB"),
    ]
    store.add(nodes)

    assert store.get_vector_count() == 3

    # Delete by node IDs
    store.delete_nodes(node_ids=["1", "2"])

    assert store.get_vector_count() == 1

    from llama_index.core.vector_stores.types import VectorStoreQuery

    res = store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=10)
    )
    assert set(res.ids) == {"3"}  # only docB remains


def test_clear(ZeusDBVectorStore):
    store = ZeusDBVectorStore(dim=3)
    store.add([FakeNode("a", [1.0, 0.0, 0.0], {}, id_="1", ref_doc_id="docA")])
    assert store.get_vector_count() == 1
    store.clear()
    assert store.get_vector_count() == 0


def test_persist_and_load(ZeusDBVectorStore, tmp_path):
    store = ZeusDBVectorStore(dim=3)
    store.add([FakeNode("a", [1.0, 0.0, 0.0], {}, id_="1", ref_doc_id="docA")])
    out = tmp_path / "index.zeusdb"
    store.persist(str(out))

    # load_index should construct via VectorDatabase.load(...)
    loaded = ZeusDBVectorStore.load_index(str(out))
    assert loaded is not None
    assert hasattr(loaded, "client")


def test_from_nodes_infers_dim(ZeusDBVectorStore):
    nodes = [
        FakeNode("a", [1.0, 0.0, 0.0, 0.0], {}, id_="1", ref_doc_id="docA"),
        FakeNode("b", [0.0, 1.0, 0.0, 0.0], {}, id_="2", ref_doc_id="docA"),
    ]
    store = ZeusDBVectorStore.from_nodes(nodes)
    assert store.get_vector_count() == 2


def test_getters(ZeusDBVectorStore):
    store = ZeusDBVectorStore(dim=2)
    store.add([FakeNode("a", [1.0, 0.0], {"tier": 1}, id_="1", ref_doc_id="d")])
    assert store.get_vector_count() == 1
    stats = store.get_zeusdb_stats()
    assert "dim" in stats and "space" in stats and "count" in stats


@pytest.mark.asyncio
async def test_async_paths(ZeusDBVectorStore):
    store = ZeusDBVectorStore(dim=3)

    # inherited BasePydanticVectorStore.async_add -> calls add() sync by default
    ids = await store.async_add(
        [FakeNode("a", [1.0, 0.0, 0.0], {}, id_="1", ref_doc_id="d")]
    )
    assert ids == ["1"]

    q = [1.0, 0.0, 0.0]
    from llama_index.core.vector_stores.types import VectorStoreQuery

    res = await store.aquery(VectorStoreQuery(query_embedding=q, similarity_top_k=1))
    assert res.ids and res.ids[0] == "1"

    # adelete by ref_doc_id should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        await store.adelete(ref_doc_id="d")

    # But adelete_nodes by ID should work
    await store.adelete_nodes(node_ids=["1"])
    res2 = await store.aquery(VectorStoreQuery(query_embedding=q, similarity_top_k=1))
    assert res2.ids == [] or res2.ids is None

    # clear via aclear should not error
    await store.aclear()


def test_query_with_mmr(ZeusDBVectorStore):
    store = ZeusDBVectorStore(dim=3)
    # three "near-ish" points around e1
    nodes = [
        FakeNode("n1", [1.0, 0.0, 0.0], {}, id_="1", ref_doc_id="d"),
        FakeNode("n2", [0.9, 0.1, 0.0], {}, id_="2", ref_doc_id="d"),
        FakeNode("n3", [0.8, 0.2, 0.0], {}, id_="3", ref_doc_id="d"),
    ]
    store.add(nodes)

    from llama_index.core.vector_stores.types import VectorStoreQuery

    # request k=2 with MMR enabled; fetch_k larger to force rerank variety
    res = store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=2),
        mmr=True,
        fetch_k=5,
        mmr_lambda=0.7,
        return_vector=True,
    )
    assert len(res.ids) == 2
    assert set(res.ids).issubset({"1", "2", "3"})
