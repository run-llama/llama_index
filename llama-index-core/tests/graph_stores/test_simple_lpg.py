import os
import tempfile

import pytest

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


def test_upsert_relation_updates_properties() -> None:
    g = SimplePropertyGraphStore()
    e1 = EntityNode(name="Alice")
    e2 = EntityNode(name="Bob")
    g.upsert_nodes([e1, e2])

    for confidence in (0.1, 0.9):
        g.upsert_relations(
            [
                Relation(
                    label="KNOWS",
                    source_id=e1.id,
                    target_id=e2.id,
                    properties={"confidence": confidence},
                )
            ]
        )

    triplets = g.get_triplets(relation_names=["KNOWS"])

    assert len(triplets) == 1
    assert triplets[0][1].properties == {"confidence": 0.9}


@pytest.mark.asyncio
async def test_aupsert_relation_updates_properties() -> None:
    g = SimplePropertyGraphStore()
    e1 = EntityNode(name="Alice")
    e2 = EntityNode(name="Bob")
    await g.aupsert_nodes([e1, e2])

    for confidence in (0.1, 0.9):
        await g.aupsert_relations(
            [
                Relation(
                    label="KNOWS",
                    source_id=e1.id,
                    target_id=e2.id,
                    properties={"confidence": confidence},
                )
            ]
        )

    triplets = g.get_triplets(relation_names=["KNOWS"])

    assert len(triplets) == 1
    assert triplets[0][1].properties == {"confidence": 0.9}


def test_upsert_relation_keeps_other_triplets_and_properties() -> None:
    g = SimplePropertyGraphStore()
    e1 = EntityNode(name="Alice", properties={"role": "analyst"})
    e2 = EntityNode(name="Bob")
    e3 = EntityNode(name="Carol")
    g.upsert_nodes([e1, e2, e3])

    knows = Relation(label="KNOWS", source_id=e1.id, target_id=e2.id)
    works_with = Relation(label="WORKS_WITH", source_id=e1.id, target_id=e3.id)
    g.upsert_relations([knows, works_with])

    updated_knows = Relation(
        label="KNOWS",
        source_id=e1.id,
        target_id=e2.id,
        properties={"confidence": 0.9},
    )
    g.upsert_relations([updated_knows])

    triplets = g.graph.get_triplets()
    relations_by_label = {t[1].label: t[1] for t in triplets}

    assert len(triplets) == 2
    assert relations_by_label["KNOWS"].properties == {"confidence": 0.9}
    assert relations_by_label["WORKS_WITH"].properties == {}
    assert g.get(ids=[e1.id])[0].properties == {"role": "analyst"}
