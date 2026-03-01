"""Integration tests for ArcadeDBPropertyGraphStore.

Requires a running ArcadeDB instance with Bolt enabled.
Set ARCADEDB_BOLT_URL (default bolt://localhost:7687) to run.

    ARCADEDB_BOLT_URL=bolt://localhost:7687 pytest tests/test_pg_stores_arcadedb.py -v
"""

from __future__ import annotations

import os

import pytest

from llama_index.core.graph_stores.types import (
    ChunkNode,
    EntityNode,
    Relation,
)
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore

BOLT_URL = os.environ.get("ARCADEDB_BOLT_URL")

pytestmark = pytest.mark.skipif(
    BOLT_URL is None,
    reason="ARCADEDB_BOLT_URL not set — skipping integration tests",
)


@pytest.fixture(scope="module")
def store():
    """Create a store connected to ArcadeDB and clean up after tests."""
    s = ArcadeDBPropertyGraphStore(
        url=BOLT_URL,
        username=os.environ.get("ARCADEDB_USERNAME", "root"),
        password=os.environ.get("ARCADEDB_PASSWORD", "playwithdata"),
        database=os.environ.get("ARCADEDB_DATABASE", ""),
    )
    yield s
    # Clean up all test data
    try:
        s.structured_query("MATCH (n) DETACH DELETE n")
    except Exception:
        pass
    s.close()


@pytest.fixture(autouse=True)
def _clean(store):
    """Clean the graph before each test."""
    try:
        store.structured_query("MATCH (n) DETACH DELETE n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Upsert & Get
# ---------------------------------------------------------------------------


class TestUpsertAndGet:
    def test_upsert_entity_nodes(self, store):
        nodes = [
            EntityNode(name="Alice", label="PERSON", properties={"age": 30}),
            EntityNode(name="Bob", label="PERSON", properties={"age": 25}),
        ]
        store.upsert_nodes(nodes)

        result = store.get(ids=["Alice", "Bob"])
        assert len(result) == 2
        names = {n.name for n in result if isinstance(n, EntityNode)}
        assert names == {"Alice", "Bob"}

    def test_upsert_chunk_nodes(self, store):
        chunks = [
            ChunkNode(id_="c1", text="The sky is blue."),
            ChunkNode(id_="c2", text="Water is wet."),
        ]
        store.upsert_nodes(chunks)

        result = store.get(ids=["c1", "c2"])
        assert len(result) == 2
        assert all(isinstance(n, ChunkNode) for n in result)

    def test_upsert_entity_idempotent(self, store):
        entity = EntityNode(name="Alice", label="PERSON", properties={"age": 30})
        store.upsert_nodes([entity])
        store.upsert_nodes([entity])  # Should not create duplicate

        result = store.get(ids=["Alice"])
        assert len(result) == 1

    def test_get_by_properties(self, store):
        store.upsert_nodes([
            EntityNode(name="Alice", label="PERSON", properties={"age": 30}),
            EntityNode(name="Bob", label="PERSON", properties={"age": 25}),
        ])

        result = store.get(properties={"age": 30})
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


class TestRelations:
    def test_upsert_relations(self, store):
        store.upsert_nodes([
            EntityNode(name="Alice", label="PERSON"),
            EntityNode(name="Bob", label="PERSON"),
        ])

        rel = Relation(source_id="Alice", target_id="Bob", label="KNOWS")
        store.upsert_relations([rel])

        triplets = store.get_triplets(entity_names=["Alice"])
        assert len(triplets) >= 1

        found = False
        for src, r, tgt in triplets:
            if r.label == "KNOWS":
                found = True
                break
        assert found, "KNOWS relationship not found in triplets"


# ---------------------------------------------------------------------------
# Triplets
# ---------------------------------------------------------------------------


class TestTriplets:
    def test_get_triplets_by_entity_name(self, store):
        store.upsert_nodes([
            EntityNode(name="Alice", label="PERSON"),
            EntityNode(name="London", label="CITY"),
        ])
        store.upsert_relations([
            Relation(source_id="Alice", target_id="London", label="LIVES_IN"),
        ])

        triplets = store.get_triplets(entity_names=["Alice"])
        assert len(triplets) >= 1
        labels = {t[1].label for t in triplets}
        assert "LIVES_IN" in labels

    def test_get_triplets_with_relation_filter(self, store):
        store.upsert_nodes([
            EntityNode(name="Alice", label="PERSON"),
            EntityNode(name="Bob", label="PERSON"),
            EntityNode(name="London", label="CITY"),
        ])
        store.upsert_relations([
            Relation(source_id="Alice", target_id="Bob", label="KNOWS"),
            Relation(source_id="Alice", target_id="London", label="LIVES_IN"),
        ])

        triplets = store.get_triplets(
            entity_names=["Alice"], relation_names=["KNOWS"]
        )
        labels = {t[1].label for t in triplets}
        assert "KNOWS" in labels
        assert "LIVES_IN" not in labels


# ---------------------------------------------------------------------------
# Rel Map
# ---------------------------------------------------------------------------


class TestRelMap:
    def test_get_rel_map(self, store):
        store.upsert_nodes([
            EntityNode(name="Alice", label="PERSON"),
            EntityNode(name="Bob", label="PERSON"),
            EntityNode(name="Charlie", label="PERSON"),
        ])
        store.upsert_relations([
            Relation(source_id="Alice", target_id="Bob", label="KNOWS"),
            Relation(source_id="Bob", target_id="Charlie", label="KNOWS"),
        ])

        alice_node = EntityNode(name="Alice", label="PERSON")
        triples = store.get_rel_map([alice_node], depth=2, limit=30)
        assert len(triples) >= 1


# ---------------------------------------------------------------------------
# Structured Query
# ---------------------------------------------------------------------------


class TestStructuredQuery:
    def test_cypher_passthrough(self, store):
        store.upsert_nodes([EntityNode(name="Test", label="THING")])
        result = store.structured_query(
            "MATCH (n:Entity) WHERE n.name = $name RETURN n.name AS name",
            param_map={"name": "Test"},
        )
        assert len(result) == 1
        assert result[0]["name"] == "Test"


# ---------------------------------------------------------------------------
# Vector Query
# ---------------------------------------------------------------------------


class TestVectorQuery:
    def test_vector_query(self, store):
        store.upsert_nodes([
            EntityNode(
                name="vec1",
                label="THING",
                embedding=[1.0, 0.0, 0.0],
            ),
            EntityNode(
                name="vec2",
                label="THING",
                embedding=[0.0, 1.0, 0.0],
            ),
        ])

        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0],
            similarity_top_k=2,
        )
        nodes, scores = store.vector_query(query)
        # At minimum the brute-force fallback should work
        assert len(nodes) >= 1
        assert nodes[0].name == "vec1"


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_by_entity_names(self, store):
        store.upsert_nodes([EntityNode(name="Alice", label="PERSON")])
        store.delete(entity_names=["Alice"])
        result = store.get(ids=["Alice"])
        assert len(result) == 0

    def test_delete_by_ids(self, store):
        store.upsert_nodes([EntityNode(name="Bob", label="PERSON")])
        store.delete(ids=["Bob"])
        result = store.get(ids=["Bob"])
        assert len(result) == 0

    def test_delete_relations(self, store):
        store.upsert_nodes([
            EntityNode(name="A", label="THING"),
            EntityNode(name="B", label="THING"),
        ])
        store.upsert_relations([
            Relation(source_id="A", target_id="B", label="LINKS_TO"),
        ])
        store.delete(relation_names=["LINKS_TO"])
        triplets = store.get_triplets(entity_names=["A"])
        links = [t for t in triplets if t[1].label == "LINKS_TO"]
        assert len(links) == 0


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_refresh_schema(self, store):
        store.upsert_nodes([
            EntityNode(name="Alice", label="PERSON", properties={"age": 30}),
        ])
        store.refresh_schema()
        schema = store.get_schema()
        assert "node_props" in schema
        # Entity type should appear
        assert "Entity" in schema["node_props"]

    def test_get_schema_str(self, store):
        store.upsert_nodes([EntityNode(name="Alice", label="PERSON")])
        store.refresh_schema()
        schema_str = store.get_schema_str()
        assert "Node properties:" in schema_str
