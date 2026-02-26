"""Unit tests for ArcadeDBPropertyGraphStore (mocked neo4j driver)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.graph_stores.types import (
    ChunkNode,
    EntityNode,
    Relation,
)
from llama_index.core.vector_stores.types import VectorStoreQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(**kwargs):
    """Create a store with a mocked driver (no real ArcadeDB needed)."""
    with patch(
        "llama_index.graph_stores.arcadedb.arcadedb_property_graph.GraphDatabase"
    ) as mock_gd:
        mock_driver = MagicMock()
        mock_gd.driver.return_value = mock_driver

        # execute_query returns (records, summary, keys) — default empty
        mock_driver.execute_query.return_value = ([], None, [])

        store = (
            __import__(
                "llama_index.graph_stores.arcadedb.arcadedb_property_graph",
                fromlist=["ArcadeDBPropertyGraphStore"],
            )
        ).ArcadeDBPropertyGraphStore(
            url="bolt://localhost:7687",
            username="root",
            password="test",
            database="testdb",
            refresh_schema=False,
            **kwargs,
        )
        return store, mock_driver


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_creates_driver_and_verifies_connectivity(self):
        store, driver = _make_store()
        driver.verify_connectivity.assert_called_once()

    def test_ensure_types_called(self):
        store, driver = _make_store()
        # _ensure_types runs 4 statements: 2 CREATE TYPE + 2 CREATE INDEX
        calls = driver.execute_query.call_args_list
        type_calls = [
            c for c in calls if "CREATE VERTEX TYPE" in str(c) or "CREATE INDEX" in str(c)
        ]
        assert len(type_calls) == 4

    def test_client_property(self):
        store, driver = _make_store()
        assert store.client is driver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_clean_properties(self):
        from llama_index.graph_stores.arcadedb.arcadedb_property_graph import (
            _clean_properties,
        )

        props = {
            "@rid": "#1:0",
            "@type": "Entity",
            "@cat": "v",
            "name": "Alice",
            "age": 30,
            "empty": "",
            "none_val": None,
        }
        cleaned = _clean_properties(props)
        assert "@rid" not in cleaned
        assert "@type" not in cleaned
        assert "@cat" not in cleaned
        assert "empty" not in cleaned
        assert "none_val" not in cleaned
        assert cleaned["name"] == "Alice"
        assert cleaned["age"] == 30

    def test_strip_embedding(self):
        from llama_index.graph_stores.arcadedb.arcadedb_property_graph import (
            _strip_embedding,
        )

        props = {"name": "Alice", "embedding": [0.1, 0.2]}
        stripped = _strip_embedding(props)
        assert "embedding" not in stripped
        assert stripped["name"] == "Alice"

    def test_infer_type(self):
        from llama_index.graph_stores.arcadedb.arcadedb_property_graph import (
            _infer_type,
        )

        assert _infer_type(42) == "INTEGER"
        assert _infer_type(3.14) == "FLOAT"
        assert _infer_type("hello") == "STRING"
        assert _infer_type(True) == "BOOLEAN"
        assert _infer_type([1, 2]) == "LIST"
        assert _infer_type(object()) == "STRING"

    def test_cosine_similarity(self):
        from llama_index.graph_stores.arcadedb.arcadedb_property_graph import (
            _cosine_similarity,
        )

        assert _cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
        assert _cosine_similarity([0, 0], [1, 0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Upsert Nodes
# ---------------------------------------------------------------------------


class TestUpsertNodes:
    def test_upsert_entity_node(self):
        store, driver = _make_store()
        driver.execute_query.reset_mock()

        entity = EntityNode(name="Alice", label="PERSON", properties={"age": 30})
        store.upsert_nodes([entity])

        calls = driver.execute_query.call_args_list
        # Should have at least one MERGE call for Entity
        merge_calls = [c for c in calls if "MERGE" in str(c) and "Entity" in str(c)]
        assert len(merge_calls) >= 1

    def test_upsert_chunk_node(self):
        store, driver = _make_store()
        driver.execute_query.reset_mock()

        chunk = ChunkNode(id_="chunk1", text="Hello world")
        store.upsert_nodes([chunk])

        calls = driver.execute_query.call_args_list
        merge_calls = [c for c in calls if "MERGE" in str(c) and "Chunk" in str(c)]
        assert len(merge_calls) >= 1

    def test_upsert_entity_with_embedding(self):
        store, driver = _make_store()
        driver.execute_query.reset_mock()

        entity = EntityNode(
            name="Alice", label="PERSON", embedding=[0.1, 0.2, 0.3]
        )
        store.upsert_nodes([entity])

        calls = driver.execute_query.call_args_list
        embedding_calls = [c for c in calls if "embedding" in str(c)]
        assert len(embedding_calls) >= 1

    def test_upsert_entity_with_triplet_source(self):
        store, driver = _make_store()
        driver.execute_query.reset_mock()

        entity = EntityNode(
            name="Alice",
            label="PERSON",
            properties={"triplet_source_id": "chunk1"},
        )
        store.upsert_nodes([entity])

        calls = driver.execute_query.call_args_list
        mentions_calls = [c for c in calls if "MENTIONS" in str(c)]
        assert len(mentions_calls) == 1


# ---------------------------------------------------------------------------
# Upsert Relations
# ---------------------------------------------------------------------------


class TestUpsertRelations:
    def test_upsert_relation(self):
        store, driver = _make_store()
        driver.execute_query.reset_mock()

        rel = Relation(source_id="Alice", target_id="Bob", label="KNOWS")
        store.upsert_relations([rel])

        calls = driver.execute_query.call_args_list
        merge_calls = [c for c in calls if "MERGE" in str(c) and "KNOWS" in str(c)]
        assert len(merge_calls) == 1


# ---------------------------------------------------------------------------
# Get Nodes
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_by_ids(self):
        store, driver = _make_store()

        # Mock response: an entity node
        mock_record = MagicMock()
        mock_record.__iter__ = MagicMock(return_value=iter([]))
        mock_record.items.return_value = [
            ("name", "Alice"),
            ("node_type", "Entity"),
            ("label", "PERSON"),
            ("props", {"id": "Alice", "name": "Alice", "label": "PERSON"}),
        ]
        mock_record.__getitem__ = lambda self, k: dict(self.items())[k]
        mock_record.get = lambda k, d=None: dict(mock_record.items()).get(k, d)
        mock_record.keys.return_value = ["name", "node_type", "label", "props"]

        driver.execute_query.return_value = ([mock_record], None, [])

        nodes = store.get(ids=["Alice"])
        assert len(nodes) == 1
        assert isinstance(nodes[0], EntityNode)

    def test_get_by_properties(self):
        store, driver = _make_store()
        driver.execute_query.return_value = ([], None, [])

        nodes = store.get(properties={"age": 30})
        assert nodes == []

        # Verify the query contained WHERE clause with property filter
        call_args = driver.execute_query.call_args_list[-1]
        query = call_args[0][0]
        assert "prop_0" in query


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_by_entity_names(self):
        store, driver = _make_store()
        driver.execute_query.reset_mock()

        store.delete(entity_names=["Alice", "Bob"])

        calls = driver.execute_query.call_args_list
        delete_calls = [c for c in calls if "DETACH DELETE" in str(c)]
        assert len(delete_calls) == 1

    def test_delete_by_ids(self):
        store, driver = _make_store()
        driver.execute_query.reset_mock()

        store.delete(ids=["id1", "id2"])

        calls = driver.execute_query.call_args_list
        delete_calls = [c for c in calls if "DETACH DELETE" in str(c)]
        assert len(delete_calls) == 1

    def test_delete_relation_names(self):
        store, driver = _make_store()
        driver.execute_query.reset_mock()

        store.delete(relation_names=["KNOWS", "LIVES_IN"])

        calls = driver.execute_query.call_args_list
        delete_calls = [c for c in calls if "DELETE" in str(c)]
        assert len(delete_calls) == 2

    def test_delete_by_properties(self):
        store, driver = _make_store()
        driver.execute_query.reset_mock()

        store.delete(properties={"age": 30})

        calls = driver.execute_query.call_args_list
        delete_calls = [c for c in calls if "DETACH DELETE" in str(c)]
        assert len(delete_calls) == 1


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_get_schema_returns_dict(self):
        store, _ = _make_store()
        # Manually trigger refresh_schema so structured_schema is populated
        store.refresh_schema()
        schema = store.get_schema()
        assert isinstance(schema, dict)
        assert "node_props" in schema

    def test_get_schema_str(self):
        store, _ = _make_store()
        store.structured_schema = {
            "node_props": {
                "Entity": [{"property": "name", "type": "STRING"}],
            },
            "rel_props": {},
            "relationships": [
                {"start": "Entity", "type": "KNOWS", "end": "Entity"},
            ],
            "metadata": {"constraint": [], "index": []},
        }
        schema_str = store.get_schema_str()
        assert "Node properties:" in schema_str
        assert "Entity" in schema_str
        assert "KNOWS" in schema_str


# ---------------------------------------------------------------------------
# Structured Query
# ---------------------------------------------------------------------------


class TestStructuredQuery:
    def test_passthrough(self):
        store, driver = _make_store()
        mock_record = MagicMock()
        mock_record.items.return_value = [("count", 42)]
        mock_record.__getitem__ = lambda self, k: dict(self.items())[k]
        mock_record.keys.return_value = ["count"]
        driver.execute_query.return_value = ([mock_record], None, [])

        result = store.structured_query(
            "MATCH (n) RETURN count(n) AS count"
        )
        assert len(result) == 1

    def test_sanitize_output(self):
        store, driver = _make_store(sanitize_query_output=True)
        mock_record = MagicMock()
        mock_record.items.return_value = [
            ("name", "Alice"),
            ("embedding", [0.1, 0.2]),
        ]
        mock_record.__getitem__ = lambda self, k: dict(self.items())[k]
        mock_record.keys.return_value = ["name", "embedding"]
        driver.execute_query.return_value = ([mock_record], None, [])

        result = store.structured_query("MATCH (n) RETURN n.name AS name, n.embedding AS embedding")
        # value_sanitize should remove embedding
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Vector Query
# ---------------------------------------------------------------------------


class TestVectorQuery:
    def test_empty_embedding_returns_empty(self):
        store, _ = _make_store()
        query = VectorStoreQuery(query_embedding=None, similarity_top_k=5)
        nodes, scores = store.vector_query(query)
        assert nodes == []
        assert scores == []

    def test_brute_force_fallback(self):
        store, driver = _make_store()

        # First call (native SQL) raises, second call (fallback MATCH) returns data
        mock_entity = MagicMock()
        mock_entity.items.return_value = [
            (
                "e",
                {
                    "id": "alice",
                    "name": "Alice",
                    "label": "PERSON",
                    "embedding": [1.0, 0.0, 0.0],
                },
            ),
        ]
        mock_entity.__getitem__ = lambda self, k: dict(self.items())[k]
        mock_entity.get = lambda k, d=None: dict(mock_entity.items()).get(k, d)
        mock_entity.keys.return_value = ["e"]

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            query_str = args[0] if args else ""
            if "NEAR" in str(query_str):
                raise RuntimeError("Native vector search not available")
            if "embedding IS NOT NULL" in str(query_str):
                return ([mock_entity], None, [])
            return ([], None, [])

        driver.execute_query.side_effect = side_effect

        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0], similarity_top_k=5
        )
        nodes, scores = store.vector_query(query)
        assert len(nodes) == 1
        assert scores[0] == pytest.approx(1.0)
