from typing import Iterator

import pytest

from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.falkordb import FalkorDBGraphStore


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in FalkorDBGraphStore.__mro__]
    assert GraphStore.__name__ in names_of_base_classes


@pytest.fixture()
def graph_store(falkordb_url: str, graph_name: str) -> Iterator[FalkorDBGraphStore]:
    store = FalkorDBGraphStore(url=falkordb_url, database=graph_name)
    store.query("MATCH (n) DETACH DELETE n")
    yield store
    store.close()


def test_upsert_and_get(graph_store: FalkorDBGraphStore) -> None:
    graph_store.upsert_triplet("node1", "related_to", "node2")

    assert ["RELATED_TO", "node2"] in graph_store.get("node1")
    assert ["RELATED_TO", "node2"] in graph_store.get_rel_map(["node1"], 1)["node1"]


def test_delete_removes_orphaned_entities(graph_store: FalkorDBGraphStore) -> None:
    """Regression: `check_edges` used to be truthy for a count of 0."""
    graph_store.upsert_triplet("node1", "related_to", "node2")
    graph_store.delete("node1", "related_to", "node2")

    assert graph_store.get("node1") == []
    assert graph_store.query("MATCH (n) RETURN count(n)")[0][0] == 0


def test_delete_keeps_entities_with_remaining_edges(
    graph_store: FalkorDBGraphStore,
) -> None:
    graph_store.upsert_triplet("node1", "related_to", "node2")
    graph_store.upsert_triplet("node1", "knows", "node3")
    graph_store.delete("node1", "related_to", "node2")

    # node1 still has the `knows` edge, node2 is now orphaned
    ids = {row[0] for row in graph_store.query("MATCH (n) RETURN n.id")}
    assert ids == {"node1", "node3"}


def test_index_is_created(graph_store: FalkorDBGraphStore) -> None:
    indexes = graph_store.query("CALL db.indexes()")
    assert any(row[0] == "Entity" for row in indexes)


def test_switch_graph_creates_index(
    graph_store: FalkorDBGraphStore, graph_name: str
) -> None:
    graph_store.switch_graph(f"{graph_name}_switched")
    indexes = graph_store.query("CALL db.indexes()")
    assert any(row[0] == "Entity" for row in indexes)


def test_refresh_schema(graph_store: FalkorDBGraphStore) -> None:
    graph_store.upsert_triplet("node1", "related_to", "node2")
    schema = graph_store.get_schema(refresh=True)
    assert "RELATED_TO" in schema
