import json
import os
import tempfile
from pathlib import Path

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


def test_relation_keys_do_not_collide(tmp_path: Path) -> None:
    g = SimplePropertyGraphStore()
    relations = [
        Relation(label="C", source_id="A_B", target_id="D", properties={"fact": 1}),
        Relation(label="B", source_id="A", target_id="C_D", properties={"fact": 2}),
    ]
    g.upsert_relations(relations)

    def assert_relations(store: SimplePropertyGraphStore) -> None:
        assert len(store.graph.relations) == 2
        for relation in relations:
            triplets = store.get_triplets(ids=[relation.source_id])
            assert len(triplets) == 1
            source, actual_relation, target = triplets[0]
            assert source.id == relation.source_id
            assert actual_relation == relation
            assert target.id == relation.target_id

    assert_relations(g)
    persist_path = str(tmp_path / "graph.json")
    g.persist(persist_path)
    loaded = SimplePropertyGraphStore.from_persist_path(persist_path)
    assert_relations(loaded)

    loaded.delete(ids=["A_B"])
    assert loaded.get_triplets(ids=["A"])[0][1] == relations[1]


def test_legacy_relation_keys_load_and_accept_colliding_relation(
    tmp_path: Path,
) -> None:
    g = SimplePropertyGraphStore()
    original = Relation(label="C", source_id="A_B", target_id="D")
    g.upsert_relations([original])
    legacy_data = g.to_dict()
    legacy_data["relations"] = {"A_B_C_D": original.model_dump()}
    legacy_data["triplets"] = list(legacy_data["triplets"])
    persist_path = tmp_path / "legacy_graph.json"
    persist_path.write_text(json.dumps(legacy_data), encoding="utf-8")

    loaded = SimplePropertyGraphStore.from_persist_path(str(persist_path))
    assert loaded.get_triplets(ids=["A_B"])[0][1] == original

    added = Relation(label="B", source_id="A", target_id="C_D")
    loaded.upsert_relations([added])
    assert loaded.get_triplets(ids=["A_B"])[0][1] == original
    assert loaded.get_triplets(ids=["A"])[0][1] == added

    loaded.persist(str(persist_path))
    reloaded = SimplePropertyGraphStore.from_persist_path(str(persist_path))
    assert reloaded.get_triplets(ids=["A_B"])[0][1] == original
    assert reloaded.get_triplets(ids=["A"])[0][1] == added
