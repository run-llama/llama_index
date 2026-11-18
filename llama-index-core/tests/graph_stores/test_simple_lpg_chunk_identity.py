from pathlib import Path
from typing import Optional

import pytest

import llama_index.core.graph_stores.types as graph_types
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation


@pytest.mark.parametrize("explicit_id", [None, "", "explicit-chunk"])
def test_loaded_chunk_keeps_persisted_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, explicit_id: Optional[str]
) -> None:
    monkeypatch.setattr(graph_types, "hash", lambda _: 111, raising=False)
    chunk = ChunkNode(text="A persisted chunk.", id_=explicit_id)
    entity = EntityNode(name="Alice")
    relation = Relation(label="MENTIONS", source_id=chunk.id, target_id=entity.id)
    persisted_id = chunk.id
    store = SimplePropertyGraphStore()
    store.upsert_nodes([chunk, entity])
    store.upsert_relations([relation])
    path = str(tmp_path / "graph.json")
    store.persist(path)

    # Emulate a new process whose built-in string hash uses a different seed.
    monkeypatch.setattr(graph_types, "hash", lambda _: 222)
    loaded = SimplePropertyGraphStore.from_persist_path(path)
    loaded_chunk = loaded.graph.nodes[persisted_id]

    assert loaded_chunk.id == persisted_id
    assert loaded.get(ids=[persisted_id]) == [loaded_chunk]
    assert loaded.get_triplets(ids=[persisted_id]) == [(loaded_chunk, relation, entity)]
    assert loaded.graph.nodes[loaded_chunk.id] is loaded_chunk

    loaded.delete(ids=[loaded_chunk.id])
    assert not loaded.graph.triplets
    assert persisted_id not in loaded.graph.nodes
