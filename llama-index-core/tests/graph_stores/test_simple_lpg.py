from llama_index.core.graph_stores.simple_labelled import SimpleLPGStore
from llama_index.core.graph_stores.types import (
    Entity,
    Relation,
)
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo


def test_add() -> None:
    g = SimpleLPGStore()

    e1 = Entity(text="e1")
    e2 = Entity(text="e2")
    r = Relation(text="r")

    g.upsert_triplets([(e1, r, e2)])

    assert len(g.graph.get_triplets()) == 1


def test_delete() -> None:
    g = SimpleLPGStore()

    e1 = Entity(text="e1")
    e2 = Entity(text="e2")
    r = Relation(text="r")

    g.upsert_triplets([(e1, r, e2)])
    g.delete([e1.text])

    assert len(g.graph.get_triplets()) == 0


def test_get() -> None:
    g = SimpleLPGStore()

    e1 = Entity(text="e1")
    e2 = Entity(text="e2", properties={"key": "value"})
    r = Relation(text="r")

    g.upsert_triplets([(e1, r, e2)])

    assert g.get() == [(e1, r, e2)]
    assert g.get(entity_names=["e1"]) == [(e1, r, e2)]
    assert g.get(entity_names=["e2"]) == [(e1, r, e2)]
    assert g.get(relation_names=["r"]) == [(e1, r, e2)]
    assert g.get(properties={"key": "value"}) == [(e1, r, e2)]


def test_add_node() -> None:
    g = SimpleLPGStore()

    n1 = TextNode(id_="n1", text="n1")
    n2 = TextNode(id_="n2", text="n2")

    g.upsert_nodes([n1, n2])

    assert len(g.graph.get_all_entities()) == 2


def test_delete_node_by_node_ids() -> None:
    g = SimpleLPGStore()

    n1 = TextNode(id_="n1", text="n1")
    n2 = TextNode(id_="n2", text="n2")

    g.upsert_nodes([n1, n2])
    g.delete_nodes(node_ids=["n1"])

    assert len(g.graph.get_all_entities()) == 1


def test_delete_node_by_ref_doc_ids() -> None:
    g = SimpleLPGStore()

    n1 = TextNode(
        id_="n1",
        text="n1",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
    )
    n2 = TextNode(id_="n2", text="n2")

    g.upsert_nodes([n1, n2])
    g.delete_nodes(ref_doc_ids=["n2"])

    assert len(g.graph.get_all_entities()) == 0

    g = SimpleLPGStore()

    n1 = TextNode(
        id_="n1",
        text="n1",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
    )
    n2 = TextNode(id_="n2", text="n2")

    g.upsert_nodes([n1, n2])
    g.delete_nodes(ref_doc_ids=["n3"])

    assert len(g.graph.get_all_entities()) == 1


def test_get_nodes() -> None:
    g = SimpleLPGStore()

    n1 = TextNode(id_="n1", text="n1")
    n2 = TextNode(id_="n2", text="n2")

    g.upsert_nodes([n1, n2])

    assert g.get_nodes(["n1", "n2"]) == [n1, n2]
