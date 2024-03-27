from llama_index.core.graph_stores.simple_labelled import SimpleLPGStore
from llama_index.core.graph_stores.types import (
    Entity,
    Relation,
)


def test_add() -> None:
    g = SimpleLPGStore()

    e1 = Entity(name="e1")
    e2 = Entity(name="e2")
    r = Relation(name="r")

    g.upsert_triplets([(e1, r, e2)])

    assert len(g.graph.get_triplets()) == 1


def test_delete() -> None:
    g = SimpleLPGStore()

    e1 = Entity(name="e1")
    e2 = Entity(name="e2")
    r = Relation(name="r")

    g.upsert_triplets([(e1, r, e2)])
    g.delete([e1.name])

    assert len(g.graph.get_triplets()) == 0


def test_get() -> None:
    g = SimpleLPGStore()

    e1 = Entity(name="e1")
    e2 = Entity(name="e2", properties={"key": "value"})
    r = Relation(name="r")

    g.upsert_triplets([(e1, r, e2)])

    assert g.get() == [(e1, r, e2)]
    assert g.get(entity_names=["e1"]) == [(e1, r, e2)]
    assert g.get(entity_names=["e2"]) == [(e1, r, e2)]
    assert g.get(relation_names=["r"]) == [(e1, r, e2)]
    assert g.get(properties={"key": "value"}) == [(e1, r, e2)]
