import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation


@pytest.mark.parametrize("json_round_trip", [False, True])
def test_from_dict_preserves_reusable_snapshot(json_round_trip: bool) -> None:
    store = SimplePropertyGraphStore()
    entity = EntityNode(name="Alice", properties={"roles": ["author"]})
    chunk = ChunkNode(id_="chunk", text="Alice wrote this.", embedding=[1.0, 0.0])
    relation = Relation(label="AUTHORED", source_id=entity.id, target_id=chunk.id)
    store.upsert_nodes([entity, chunk])
    store.upsert_relations([relation])
    data = (
        json.loads(store.graph.model_dump_json())
        if json_round_trip
        else store.to_dict()
    )
    original = deepcopy(data)

    first = SimplePropertyGraphStore.from_dict(data)
    second = SimplePropertyGraphStore.from_dict(data)

    assert data == original
    for loaded in (first, second):
        assert loaded.get(ids=[entity.id, chunk.id]) == [entity, chunk]
        assert loaded.get_triplets(ids=[entity.id]) == [(entity, relation, chunk)]
        assert loaded.to_dict() == store.to_dict()


def test_from_dict_validation_failure_preserves_snapshot() -> None:
    data = {
        "nodes": {"Alice": {"name": "Alice"}},
        "relations": {"invalid": {"source_id": "Alice", "target_id": "Alice"}},
        "triplets": [],
    }
    original = deepcopy(data)

    with pytest.raises(ValidationError):
        SimplePropertyGraphStore.from_dict(data)

    assert data == original
