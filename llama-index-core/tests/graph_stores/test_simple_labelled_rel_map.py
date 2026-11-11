import pytest

from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, Relation


def _make_store(edges):
    store = SimplePropertyGraphStore()
    store.upsert_relations(
        [
            Relation(source_id=source, label=label, target_id=target)
            for source, label, target in edges
        ]
    )
    return store


def _triplet_ids(triplets):
    return [
        (source.id, relation.id, target.id) for source, relation, target in triplets
    ]


@pytest.mark.parametrize("depth", [2, 3, 8])
def test_rel_map_does_not_repeat_first_hop(depth):
    store = _make_store([("A", "R", "B")])
    result = store.get_rel_map([EntityNode(name="A")], depth=depth)
    assert _triplet_ids(result) == [("A", "R", "B")]


def test_rel_map_limit_counts_distinct_relationships():
    edges = [("A", "R", "B"), ("B", "R", "C"), ("C", "R", "D")]
    store = _make_store(edges)
    result = store.get_rel_map([EntityNode(name="A")], depth=3, limit=3)
    # A repeated first-hop edge must not displace the evidence at depth three.
    assert _triplet_ids(result) == edges


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_rel_map_preserves_depth_bound(depth):
    edges = [("A", "R", "B"), ("B", "R", "C"), ("C", "R", "D")]
    store = _make_store(edges)
    result = store.get_rel_map([EntityNode(name="A")], depth=depth)
    assert _triplet_ids(result) == edges[:depth]


@pytest.mark.parametrize("limit", [0, 1, 2, 3])
def test_rel_map_preserves_limit(limit):
    edges = [("A", "R", "B"), ("B", "R", "C"), ("C", "R", "D")]
    store = _make_store(edges)
    result = store.get_rel_map([EntityNode(name="A")], depth=3, limit=limit)
    assert _triplet_ids(result) == edges[:limit]


def test_rel_map_cycle_and_self_loop_are_unique():
    edges = [("A", "SELF", "A"), ("A", "R", "B"), ("B", "R", "A")]
    store = _make_store(edges)
    result = _triplet_ids(store.get_rel_map([EntityNode(name="A")], depth=10))
    assert len(result) == len(edges)
    assert set(result) == set(edges)


def test_rel_map_does_not_merge_distinct_relation_labels():
    edges = [("A", "SUPPORTS", "B"), ("A", "CONTRADICTS", "B")]
    store = _make_store(edges)
    result = _triplet_ids(store.get_rel_map([EntityNode(name="A")], depth=3))
    assert len(result) == 2
    assert set(result) == set(edges)


def test_rel_map_overlapping_seeds_do_not_duplicate_evidence():
    edges = [("A", "R", "B"), ("B", "R", "C"), ("C", "R", "D")]
    store = _make_store(edges)
    result = _triplet_ids(
        store.get_rel_map([EntityNode(name="A"), EntityNode(name="B")], depth=3)
    )
    assert len(result) == len(edges)
    assert set(result) == set(edges)


def test_rel_map_ignored_relations_do_not_consume_result_limit():
    store = _make_store([("A", "IGNORE", "B"), ("B", "KEEP", "C")])
    result = store.get_rel_map(
        [EntityNode(name="A")], depth=3, limit=1, ignore_rels=["IGNORE"]
    )
    assert _triplet_ids(result) == [("B", "KEEP", "C")]


def test_rel_map_empty_seed_and_missing_seed():
    store = _make_store([("A", "R", "B")])
    assert store.get_rel_map([]) == []
    assert store.get_rel_map([EntityNode(name="missing")]) == []


def test_rel_map_preserves_properties_without_mutating_graph():
    store = SimplePropertyGraphStore()
    store.upsert_nodes([EntityNode(name="A", properties={"source_id": "doc-1"})])
    store.upsert_relations(
        [
            Relation(
                source_id="A",
                label="R",
                target_id="B",
                properties={"source_id": "doc-2"},
            )
        ]
    )
    before = store.to_dict()
    result = store.get_rel_map([EntityNode(name="A")], depth=3)
    assert len(result) == 1
    assert result[0][0].properties == {"source_id": "doc-1"}
    assert result[0][1].properties == {"source_id": "doc-2"}
    assert store.to_dict() == before


def test_rel_map_after_persist_reload(tmp_path):
    edges = [("A", "R", "B"), ("B", "R", "C"), ("C", "R", "D")]
    store = _make_store(edges)
    path = str(tmp_path / "graph.json")
    store.persist(path)
    restored = SimplePropertyGraphStore.from_persist_path(path)
    result = restored.get_rel_map([EntityNode(name="A")], depth=3, limit=3)
    assert _triplet_ids(result) == edges
