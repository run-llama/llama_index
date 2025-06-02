import pytest
import os

from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore

neo4j_url = os.environ.get("NEO4J_URI")
neo4j_user = os.environ.get("NEO4J_USERNAME")
neo4j_password = os.environ.get("NEO4J_PASSWORD")

if not neo4j_url or not neo4j_user or not neo4j_password:
    neo4j_available = False
else:
    neo4j_available = True

pytestmark = pytest.mark.skipif(
    not neo4j_available,
    reason="Requires NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD environment variables.",
)


@pytest.fixture()
def neo4j_graph_store() -> Neo4jGraphStore:
    """
    Provides a fresh Neo4jGraphStore for each test.
    Adjust parameters to match your test database or local Neo4j setup.
    """
    neo4j_graph_store = Neo4jGraphStore(
        username=neo4j_user,
        password=neo4j_password,
        url=neo4j_url,
        refresh_schema=True,
    )
    # Teardown: remove any remaining data to avoid polluting tests
    # For a small test DB, you can delete all nodes & relationships:
    neo4j_graph_store.query("MATCH (n) DETACH DELETE n")
    return neo4j_graph_store


def test_neo4j_graph_store():
    names_of_bases = [b.__name__ for b in Neo4jGraphStore.__bases__]
    assert GraphStore.__name__ in names_of_bases


def test_01_connection_init(neo4j_graph_store: Neo4jGraphStore):
    """
    Test initial connection and constraint creation.
    Verifies that the store is connected and schema can be fetched.
    """
    assert neo4j_graph_store is not None
    schema_str = neo4j_graph_store.get_schema(refresh=True)
    # We don't necessarily expect non-empty schema if DB is brand-new,
    # but we at least can check that it's a string.
    assert isinstance(schema_str, str)


def test_02_upsert_and_get(neo4j_graph_store: Neo4jGraphStore):
    """
    Test inserting triplets and retrieving them.
    """
    # Insert a simple triplet: Alice -> LIKES -> IceCream
    neo4j_graph_store.upsert_triplet("Alice", "LIKES", "IceCream")

    # Retrieve edges from 'Alice'
    results = neo4j_graph_store.get("Alice")
    assert len(results) == 1
    rel_type, obj = results[0]
    assert rel_type == "LIKES"
    assert obj == "IceCream"


def test_03_upsert_multiple_and_get(neo4j_graph_store: Neo4jGraphStore):
    """
    Insert multiple triplets for a single subject.
    """
    # Add two different relationships from 'Alice'
    neo4j_graph_store.upsert_triplet("Alice", "LIKES", "IceCream")
    neo4j_graph_store.upsert_triplet("Alice", "DISLIKES", "Spinach")

    results = neo4j_graph_store.get("Alice")
    # Expect two relationships
    assert len(results) == 2
    rels = {rel[0] for rel in results}
    objs = {rel[1] for rel in results}
    assert rels == {"LIKES", "DISLIKES"}
    assert objs == {"IceCream", "Spinach"}


def test_04_get_rel_map(neo4j_graph_store: Neo4jGraphStore):
    """
    Test get_rel_map with multi-hop relationships.
    """
    # Insert:
    #   Alice -> KNOWS -> Bob -> LIVES_IN -> CityX
    #   Bob -> TRAVELED_TO -> CityY
    store = neo4j_graph_store
    store.upsert_triplet("Alice", "KNOWS", "Bob")
    store.upsert_triplet("Bob", "LIVES_IN", "CityX")
    store.upsert_triplet("Bob", "TRAVELED_TO", "CityY")

    # Depth 2 from 'Alice' should see: (KNOWS->Bob) + (Bob->LIVES_IN->CityX) + (Bob->TRAVELED_TO->CityY)
    rel_map = store.get_rel_map(["Alice"], depth=2, limit=30)

    assert "Alice" in rel_map
    # Flattened relationships are a bit tricky; we only check that something is returned
    # The structure is like lists of [relType, objectId].
    flattened_rels = rel_map["Alice"]
    assert len(flattened_rels) > 0


def test_05_delete_relationship_and_nodes(neo4j_graph_store: Neo4jGraphStore):
    """
    Test deleting an existing relationship (and subject/object if no other edges).
    """
    store = neo4j_graph_store
    store.upsert_triplet("X", "REL", "Y")

    # Confirm upsert worked
    results_before = store.get("X")
    assert len(results_before) == 1
    assert results_before[0] == ["REL", "Y"]

    # Delete that relationship
    store.delete("X", "REL", "Y")

    # Now both X and Y should be removed if no other edges remain.
    results_after = store.get("X")
    assert len(results_after) == 0


def test_06_delete_keeps_node_if_other_edges_exist(neo4j_graph_store: Neo4jGraphStore):
    """
    Test that only the specified relationship is removed,
    and the subject/object are deleted only if they have no other edges.
    """
    store = neo4j_graph_store
    # Insert two edges: X->REL->Y and X->OTHER->Z
    store.upsert_triplet("X", "REL", "Y")
    store.upsert_triplet("X", "OTHER", "Z")

    # Delete the first relationship
    store.delete("X", "REL", "Y")

    # 'Y' should be gone if no other edges, but X must remain (it still has an edge to Z).
    # Confirm X->OTHER->Z still exists
    results_x = store.get("X")
    assert len(results_x) == 1
    assert results_x[0] == ["OTHER", "Z"]

    # 'Y' should have 0 edges
    results_y = store.get("Y")
    assert len(results_y) == 0


def test_07_refresh_schema(neo4j_graph_store: Neo4jGraphStore):
    """
    Test the refresh_schema call.
    """
    store = neo4j_graph_store
    # Insert a couple triplets
    store.upsert_triplet("A", "TEST_REL", "B")
    # Refresh schema
    store.refresh_schema()
    structured = store.structured_schema
    assert "node_props" in structured
    assert "rel_props" in structured
    assert "relationships" in structured


def test_08_get_schema(neo4j_graph_store: Neo4jGraphStore):
    """
    Test get_schema with and without refresh.
    """
    store = neo4j_graph_store
    # Possibly empty if no data, but we can at least confirm it doesn't error
    schema_str_1 = store.get_schema(refresh=False)
    assert isinstance(schema_str_1, str)

    # Add data
    store.upsert_triplet("Person1", "LIKES", "Thing1")
    # Now refresh
    schema_str_2 = store.get_schema(refresh=True)
    assert isinstance(schema_str_2, str)
    # The new schema might mention 'LIKES' or node labels, but that depends on your DB.
    # You can do a substring check if you expect it:
    # assert "LIKES" in schema_str_2


def test_09_custom_query(neo4j_graph_store: Neo4jGraphStore):
    """
    Test running a direct custom Cypher query via .query().
    """
    store = neo4j_graph_store
    store.upsert_triplet("TestS", "TEST_REL", "TestO")

    # Custom query to find all nodes that have an outgoing relationship
    custom_cypher = """
    MATCH (n)-[r]->(m)
    RETURN n.id AS subject, type(r) AS relation, m.id AS object
    """
    results = store.query(custom_cypher)
    assert len(results) >= 1
    # Expect at least the one we inserted
    expected = {"subject": "TestS", "relation": "TEST_REL", "object": "TestO"}
    assert any(
        row["subject"] == expected["subject"]
        and row["relation"] == expected["relation"]
        and row["object"] == expected["object"]
        for row in results
    ), "Custom query did not return the inserted relationship."
