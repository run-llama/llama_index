import pytest

from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, Relation


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_relation_upsert_updates_existing_properties(use_async: bool) -> None:
    store = SimplePropertyGraphStore()
    alice = EntityNode(name="Alice", properties={"role": "author"})
    bob = EntityNode(name="Bob", properties={"role": "reviewer"})
    original = Relation(
        label="KNOWS",
        source_id=alice.id,
        target_id=bob.id,
        properties={"confidence": 0.1},
    )
    other = Relation(label="MENTIONS", source_id=alice.id, target_id=bob.id)
    updated = Relation(
        label="KNOWS",
        source_id=alice.id,
        target_id=bob.id,
        properties={"confidence": 0.9},
    )
    store.upsert_nodes([alice, bob])
    store.upsert_relations([original, other])

    if use_async:
        await store.aupsert_relations([updated])
    else:
        store.upsert_relations([updated])

    assert store.get_triplets(relation_names=["KNOWS"]) == [(alice, updated, bob)]
    assert store.get_triplets(relation_names=["MENTIONS"]) == [(alice, other, bob)]
    assert len(store.graph.triplets) == 2
    assert len(store.graph.relations) == 2
    assert store.get(ids=[alice.id, bob.id]) == [alice, bob]

    if use_async:
        await store.aupsert_relations([updated])
    else:
        store.upsert_relations([updated])
    assert len(store.graph.triplets) == 2
    assert store.get_triplets(relation_names=["KNOWS"]) == [(alice, updated, bob)]
