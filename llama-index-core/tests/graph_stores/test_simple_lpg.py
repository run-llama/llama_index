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
