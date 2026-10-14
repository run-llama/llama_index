import os
import tempfile

from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
)
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo


def test_add() -> None:
    g = SimplePropertyGraphStore()

    e1 = EntityNode(name="e1")
    e2 = EntityNode(name="e2")
    r = Relation(label="r", source_id=e1.id, target_id=e2.id)

    g.upsert_nodes([e1, e2])
    g.upsert_relations([r])

    assert len(g.graph.get_triplets()) == 1


def test_delete() -> None:
    g = SimplePropertyGraphStore()

    e1 = EntityNode(name="e1")
    e2 = EntityNode(name="e2")
    r = Relation(label="r", source_id=e1.id, target_id=e2.id)

    g.upsert_nodes([e1, e2])
    g.upsert_relations([r])
    g.delete(ids=[e1.id])

    assert len(g.graph.get_triplets()) == 0


def test_delete_shared_endpoint_keeps_other_triplet() -> None:
    """Deleting one endpoint of a shared node must not drop the other triplet."""
    g = SimplePropertyGraphStore()

    alice = EntityNode(name="Alice")
    bob = EntityNode(name="Bob")
    carol = EntityNode(name="Carol")
    knows = Relation(label="knows", source_id=alice.id, target_id=bob.id)
    likes = Relation(label="likes", source_id=alice.id, target_id=carol.id)

    g.upsert_nodes([alice, bob, carol])
    g.upsert_relations([knows, likes])
    g.delete(ids=["Bob"])

    assert "Alice" in g.graph.nodes
    assert "Carol" in g.graph.nodes
    assert "Bob" not in g.graph.nodes
    triplets = g.graph.get_triplets()
    assert len(triplets) == 1
    assert triplets[0][0].id == "Alice"
    assert triplets[0][1].id == "likes"
    assert triplets[0][2].id == "Carol"


def test_delete_empty_selectors_leave_graph_unchanged() -> None:
    """Empty id selectors must not wipe the graph."""
    g = SimplePropertyGraphStore()

    alice = EntityNode(name="Alice")
    bob = EntityNode(name="Bob")
    carol = EntityNode(name="Carol")
    knows = Relation(label="knows", source_id=alice.id, target_id=bob.id)
    likes = Relation(label="likes", source_id=alice.id, target_id=carol.id)

    g.upsert_nodes([alice, bob, carol])
    g.upsert_relations([knows, likes])

    g.delete(ids=[])

    assert set(g.graph.nodes) == {"Alice", "Bob", "Carol"}
    assert len(g.graph.relations) == 2
    assert len(g.graph.get_triplets()) == 2

    g.delete_llama_nodes(node_ids=["missing-id"])

    assert set(g.graph.nodes) == {"Alice", "Bob", "Carol"}
    assert len(g.graph.relations) == 2
    assert len(g.graph.get_triplets()) == 2


def test_delete_entity_names_keeps_unrelated_triplet() -> None:
    """Deleting an entity removes that entity and its triplets only."""
    g = SimplePropertyGraphStore()

    alice = EntityNode(name="Alice")
    bob = EntityNode(name="Bob")
    carol = EntityNode(name="Carol")
    dave = EntityNode(name="Dave")
    knows = Relation(label="knows", source_id=alice.id, target_id=bob.id)
    likes = Relation(label="likes", source_id=carol.id, target_id=bob.id)

    g.upsert_nodes([alice, bob, carol, dave])
    g.upsert_relations([knows, likes])
    g.delete(entity_names=["Alice"])

    assert "Alice" not in g.graph.nodes
    assert "Bob" in g.graph.nodes
    assert "Carol" in g.graph.nodes
    assert "Dave" in g.graph.nodes
    triplets = g.graph.get_triplets()
    assert len(triplets) == 1
    assert triplets[0][0].id == "Carol"
    assert triplets[0][1].id == "likes"
    assert triplets[0][2].id == "Bob"


def test_delete_relation_names_keeps_referenced_nodes() -> None:
    """Deleting a relation removes those triplets and unreferenced endpoints."""
    g = SimplePropertyGraphStore()

    alice = EntityNode(name="Alice")
    bob = EntityNode(name="Bob")
    carol = EntityNode(name="Carol")
    dave = EntityNode(name="Dave")
    knows = Relation(label="knows", source_id=alice.id, target_id=bob.id)
    likes = Relation(label="likes", source_id=alice.id, target_id=carol.id)

    g.upsert_nodes([alice, bob, carol, dave])
    g.upsert_relations([knows, likes])
    g.delete(relation_names=["knows"])

    triplets = g.graph.get_triplets()
    assert len(triplets) == 1
    assert triplets[0][0].id == "Alice"
    assert triplets[0][1].id == "likes"
    assert triplets[0][2].id == "Carol"
    # Bob was only used by "knows"; Alice and Carol are still linked.
    assert set(g.graph.nodes) == {"Alice", "Carol", "Dave"}
    assert "Alice_knows_Bob" not in g.graph.relations
    assert "Alice_likes_Carol" in g.graph.relations


def test_get() -> None:
    g = SimplePropertyGraphStore()

    e1 = EntityNode(name="e1")
    e2 = EntityNode(name="e2", properties={"key": "value"})
    r = Relation(label="r", source_id=e1.id, target_id=e2.id)

    g.upsert_nodes([e1, e2])
    g.upsert_relations([r])

    assert g.get_triplets() == []
    assert g.get_triplets(entity_names=["e1"]) == [(e1, r, e2)]
    assert g.get_triplets(entity_names=["e2"]) == [(e1, r, e2)]
    assert g.get_triplets(relation_names=["r"]) == [(e1, r, e2)]
    assert g.get_triplets(properties={"key": "value"}) == [(e1, r, e2)]


def test_add_node() -> None:
    g = SimplePropertyGraphStore()

    n1 = TextNode(id_="n1", text="n1")
    n2 = TextNode(id_="n2", text="n2")

    g.upsert_llama_nodes([n1, n2])

    assert len(g.graph.get_all_nodes()) == 2


def test_delete_node_by_node_ids() -> None:
    g = SimplePropertyGraphStore()

    n1 = TextNode(id_="n1", text="n1")
    n2 = TextNode(id_="n2", text="n2")

    g.upsert_llama_nodes([n1, n2])
    g.delete_llama_nodes(node_ids=["n1"])

    assert len(g.graph.get_all_nodes()) == 1


def test_delete_node_by_ref_doc_ids() -> None:
    g = SimplePropertyGraphStore()

    n1 = TextNode(
        id_="n1",
        text="n1",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
    )
    n2 = TextNode(id_="n2", text="n2")

    g.upsert_llama_nodes([n1, n2])
    g.delete_llama_nodes(ref_doc_ids=["n2"])

    assert len(g.graph.get_all_nodes()) == 0

    g = SimplePropertyGraphStore()

    n1 = TextNode(
        id_="n1",
        text="n1",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
    )
    n2 = TextNode(id_="n2", text="n2")

    g.upsert_llama_nodes([n1, n2])
    g.delete_llama_nodes(ref_doc_ids=["n3"])

    assert len(g.graph.get_all_nodes()) == 1


def test_get_nodes() -> None:
    g = SimplePropertyGraphStore()

    n1 = TextNode(id_="n1", text="n1")
    n2 = TextNode(id_="n2", text="n2")

    g.upsert_llama_nodes([n1, n2])

    assert g.get_llama_nodes(["n1", "n2"]) == [n1, n2]


def test_persist_utf8_round_trip() -> None:
    """Test that persist/load works with non-ASCII characters (issue #21109)."""
    g = SimplePropertyGraphStore()

    # Use Chinese, Japanese, and special Unicode characters
    e1 = EntityNode(name="定义图", properties={"desc": "中文描述"})
    e2 = EntityNode(name="テスト", properties={"desc": "日本語の説明"})
    e3 = EntityNode(name="émojis_✨🚀", properties={"desc": "spëcîal ♠♣♥♦"})
    r1 = Relation(label="关系", source_id=e1.id, target_id=e2.id)
    r2 = Relation(label="リンク", source_id=e2.id, target_id=e3.id)

    g.upsert_nodes([e1, e2, e3])
    g.upsert_relations([r1, r2])

    with tempfile.TemporaryDirectory() as tmpdir:
        persist_path = os.path.join(tmpdir, "test_graph.json")
        g.persist(persist_path)

        loaded = SimplePropertyGraphStore.from_persist_path(persist_path)

    assert len(loaded.graph.get_triplets()) == 2
    assert len(loaded.graph.get_all_nodes()) == 3

    loaded_nodes = {n.id: n for n in loaded.get(ids=[e1.id, e2.id, e3.id])}
    assert loaded_nodes[e1.id].name == "定义图"
    assert loaded_nodes[e2.id].name == "テスト"
    assert loaded_nodes[e3.id].name == "émojis_✨🚀"
