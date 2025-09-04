import os
import pytest

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, ChunkNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery


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
def neo4j_store() -> Neo4jPropertyGraphStore:
    if not neo4j_available:
        pytest.skip("No Neo4j credentials provided")
    neo4j_store = Neo4jPropertyGraphStore(
        username=neo4j_user, password=neo4j_password, url=neo4j_url
    )
    neo4j_store.structured_query("MATCH (n) DETACH DELETE n")
    return neo4j_store


def test_upsert_nodes_and_get(neo4j_store: Neo4jPropertyGraphStore):
    """
    Test inserting entity and chunk nodes, then retrieving them.
    """
    entity = EntityNode(label="PERSON", name="Alice")
    chunk = ChunkNode(text="Alice is a software engineer.")
    neo4j_store.upsert_nodes([entity, chunk])

    # Get by ID
    retrieved_entities = neo4j_store.get(ids=[entity.id])
    assert len(retrieved_entities) == 1
    assert isinstance(retrieved_entities[0], EntityNode)
    assert retrieved_entities[0].name == "Alice"

    retrieved_chunks = neo4j_store.get(ids=[chunk.id])
    assert len(retrieved_chunks) == 1
    assert isinstance(retrieved_chunks[0], ChunkNode)
    assert retrieved_chunks[0].text == "Alice is a software engineer."

    # Get by property
    retrieved_by_prop = neo4j_store.get(properties={"name": "Alice"})
    assert len(retrieved_by_prop) == 1
    assert retrieved_by_prop[0].id == entity.id

    # Attempt to get unknown property
    unknown_prop = neo4j_store.get(properties={"non_existent_prop": "foo"})
    assert len(unknown_prop) == 0


def test_02_upsert_nodes_and_get_multiple(neo4j_store):
    """
    Test inserting multiple nodes at once and retrieving them by IDs.
    """
    entity1 = EntityNode(label="PERSON", name="Bob")
    entity2 = EntityNode(label="PERSON", name="Charlie")
    chunk1 = ChunkNode(text="This is sample text.")
    chunk2 = ChunkNode(text="Another sample text.")

    # Upsert multiple
    neo4j_store.upsert_nodes([entity1, entity2, chunk1, chunk2])

    # Retrieve by IDs
    ids_to_get = [entity1.id, entity2.id, chunk1.id, chunk2.id]
    results = neo4j_store.get(ids=ids_to_get)
    assert len(results) == 4

    # Check some known values
    person_bob = [r for r in results if isinstance(r, EntityNode) and r.name == "Bob"]
    assert len(person_bob) == 1

    chunk_texts = [r for r in results if isinstance(r, ChunkNode)]
    assert len(chunk_texts) == 2


def test_03_upsert_relations_and_get(neo4j_store):
    """
    Test creating relations between nodes, then retrieving them in multiple ways.
    """
    person = EntityNode(label="PERSON", name="Alice")
    city = EntityNode(label="CITY", name="Paris")
    neo4j_store.upsert_nodes([person, city])

    # Create a relation
    visited_relation = Relation(
        source_id=person.id,
        target_id=city.id,
        label="VISITED",
        properties={"year": 2023},
    )
    neo4j_store.upsert_relations([visited_relation])

    # Validate that the relation can be found in triplets
    triplets = neo4j_store.get_triplets(entity_names=["Alice"])
    assert len(triplets) == 1
    source, rel, target = triplets[0]
    assert source.name == "Alice"
    assert target.name == "Paris"
    assert rel.label == "VISITED"
    assert rel.properties["year"] == 2023


def test_05_filter_nodes_by_property(neo4j_store):
    """
    Test get() with property filtering.
    """
    e1 = EntityNode(label="PERSON", name="Alice", properties={"country": "France"})
    e2 = EntityNode(label="PERSON", name="Bob", properties={"country": "USA"})
    e3 = EntityNode(label="PERSON", name="Charlie", properties={"country": "France"})
    neo4j_store.upsert_nodes([e1, e2, e3])

    # Filter
    filtered = neo4j_store.get(properties={"country": "France"})
    assert len(filtered) == 2
    filtered_names = {x.name for x in filtered}
    assert filtered_names == {"Alice", "Charlie"}


def test_06_delete_entities_by_names(neo4j_store):
    """
    Test deleting nodes by entity_names.
    """
    e1 = EntityNode(label="PERSON", name="Alice")
    e2 = EntityNode(label="PERSON", name="Bob")
    neo4j_store.upsert_nodes([e1, e2])

    # Delete 'Alice'
    neo4j_store.delete(entity_names=["Alice"])

    # Verify
    remaining = neo4j_store.get()
    assert len(remaining) == 1
    assert remaining[0].name == "Bob"


def test_07_delete_nodes_by_ids(neo4j_store):
    """
    Test deleting nodes by IDs.
    """
    e1 = EntityNode(label="PERSON", name="Alice")
    e2 = EntityNode(label="PERSON", name="Bob")
    e3 = EntityNode(label="PERSON", name="Charlie")
    neo4j_store.upsert_nodes([e1, e2, e3])

    # Delete Bob, Charlie by IDs
    neo4j_store.delete(ids=[e2.id, e3.id])

    all_remaining = neo4j_store.get()
    assert len(all_remaining) == 1
    assert all_remaining[0].name == "Alice"


def test_08_delete_relations(neo4j_store):
    """
    Test deleting relationships by relation names.
    """
    e1 = EntityNode(label="PERSON", name="Alice")
    e2 = EntityNode(label="CITY", name="Paris")
    neo4j_store.upsert_nodes([e1, e2])

    rel = Relation(source_id=e1.id, target_id=e2.id, label="VISITED")
    neo4j_store.upsert_relations([rel])

    # Ensure the relationship is there
    triplets_before = neo4j_store.get_triplets(entity_names=["Alice"])
    assert len(triplets_before) == 1

    # Delete the relation
    neo4j_store.delete(relation_names=["VISITED"])

    # No more triplets
    triplets_after = neo4j_store.get_triplets(entity_names=["Alice"])
    assert len(triplets_after) == 0


def test_09_delete_nodes_by_properties(neo4j_store):
    """
    Test deleting nodes by a property dict.
    """
    c1 = ChunkNode(text="This is a test chunk.", properties={"lang": "en"})
    c2 = ChunkNode(text="Another chunk.", properties={"lang": "fr"})
    neo4j_store.upsert_nodes([c1, c2])

    # Delete all English chunks
    neo4j_store.delete(properties={"lang": "en"})

    # Only c2 remains
    remaining = neo4j_store.get()
    assert len(remaining) == 1
    assert remaining[0].properties["lang"] == "fr"


def test_10_vector_query(neo4j_store):
    """
    Test vector_query with some dummy embeddings.
    Note: This requires Neo4j 5.23+ for native vector indexing,
    or it falls back to approximate "cosine" with APOC.
    """
    entity1 = EntityNode(
        label="PERSON", name="Alice", properties={"embedding": [0.1, 0.2, 0.3]}
    )
    entity2 = EntityNode(
        label="PERSON", name="Bob", properties={"embedding": [0.9, 0.8, 0.7]}
    )
    neo4j_store.upsert_nodes([entity1, entity2])

    # Query embedding somewhat closer to [0.1, 0.2, 0.3] than [0.9, 0.8, 0.7]
    query = VectorStoreQuery(query_embedding=[0.1, 0.2, 0.31], similarity_top_k=2)
    results, scores = neo4j_store.vector_query(query)

    # Expect "Alice" to come first
    assert len(results) == 2
    names_in_order = [r.name for r in results]
    assert names_in_order[0] == "Alice"
    assert names_in_order[1] == "Bob"
    # Score check: Usually Alice's score should be higher
    assert scores[0] >= scores[1]


def test_11_get_rel_map(neo4j_store):
    """
    Test get_rel_map with a multi-depth scenario.
    """
    e1 = EntityNode(label="PERSON", name="Alice")
    e2 = EntityNode(label="PERSON", name="Bob")
    e3 = EntityNode(label="CITY", name="Paris")
    e4 = EntityNode(label="CITY", name="London")
    neo4j_store.upsert_nodes([e1, e2, e3, e4])

    r1 = Relation(label="KNOWS", source_id=e1.id, target_id=e2.id)
    r2 = Relation(label="VISITED", source_id=e1.id, target_id=e3.id)
    r3 = Relation(label="VISITED", source_id=e2.id, target_id=e4.id)
    neo4j_store.upsert_relations([r1, r2, r3])

    # Depth 2 should capture up to "Alice - Bob - London" chain
    rel_map = neo4j_store.get_rel_map([e1], depth=2)
    # Expect at least 2-3 relationships
    labels_found = {trip[1].label for trip in rel_map}
    assert "KNOWS" in labels_found
    assert "VISITED" in labels_found


def test_12_get_schema(neo4j_store):
    """
    Test get_schema. The schema might be empty or minimal if no data has been inserted yet.
    """
    # Insert some data first
    e1 = EntityNode(label="PERSON", name="Alice")
    neo4j_store.upsert_nodes([e1])

    schema = neo4j_store.get_schema(refresh=True)
    assert "node_props" in schema
    assert "rel_props" in schema
    assert "relationships" in schema


def test_13_get_schema_str(neo4j_store):
    """
    Test the textual representation of the schema.
    """
    e1 = EntityNode(label="PERSON", name="Alice")
    e2 = EntityNode(label="CITY", name="Paris")
    neo4j_store.upsert_nodes([e1, e2])

    # Insert a relationship
    r = Relation(label="VISITED", source_id=e1.id, target_id=e2.id)
    neo4j_store.upsert_relations([r])

    schema_str = neo4j_store.get_schema_str(refresh=True)
    assert "PERSON" in schema_str
    assert "CITY" in schema_str
    assert "VISITED" in schema_str


def test_14_structured_query(neo4j_store):
    """
    Test running a custom Cypher query via structured_query.
    """
    # Insert data
    e1 = EntityNode(label="PERSON", name="Alice")
    neo4j_store.upsert_nodes([e1])

    # Custom query
    query = """
    MATCH (n) WHERE n.name = $name
    RETURN n.name AS node_name, labels(n) AS node_labels
    """
    result = neo4j_store.structured_query(query, {"name": "Alice"})
    assert len(result) == 1
    assert result[0]["node_name"] == "Alice"
    assert "PERSON" in result[0]["node_labels"]


def test_15_refresh_schema(neo4j_store):
    """
    Test explicit refresh of the schema.
    """
    # Insert data
    e1 = EntityNode(label="PERSON", name="Alice", properties={"age": 30})
    neo4j_store.upsert_nodes([e1])

    # Refresh schema
    neo4j_store.refresh_schema()
    schema = neo4j_store.structured_schema
    assert "node_props" in schema
    person_props = schema["node_props"].get("PERSON", [])
    prop_names = {prop["property"] for prop in person_props}
    assert "age" in prop_names, "Expected 'age' property in PERSON schema."
