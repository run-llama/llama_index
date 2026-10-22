from llama_index.core.graph_stores.simple import SimpleGraphStore


def test_delete_triplet() -> None:
    store = SimpleGraphStore()
    store.upsert_triplet("a", "likes", "b")
    store.upsert_triplet("a", "hates", "c")

    store.delete("a", "likes", "b")

    assert store.get("a") == [["hates", "c"]]


def test_delete_last_triplet_removes_subject() -> None:
    store = SimpleGraphStore()
    store.upsert_triplet("a", "likes", "b")

    store.delete("a", "likes", "b")

    assert store.get("a") == []
    assert "a" not in store.get_rel_map()


def test_delete_missing_triplet_is_noop() -> None:
    store = SimpleGraphStore()
    store.upsert_triplet("a", "likes", "b")

    store.delete("a", "likes", "x")
    store.delete("z", "likes", "b")

    assert store.get("a") == [["likes", "b"]]


def test_delete_triplet_after_persist_roundtrip(tmp_path) -> None:
    persist_path = str(tmp_path / "graph_store.json")
    store = SimpleGraphStore()
    store.upsert_triplet("a", "likes", "b")
    store.upsert_triplet("a", "likes", "b")
    store.upsert_triplet("a", "knows", "c")
    store.persist(persist_path=persist_path)

    reloaded = SimpleGraphStore.from_persist_path(persist_path=persist_path)
    reloaded.delete("a", "likes", "b")
    assert reloaded.get("a") == [["knows", "c"]]

