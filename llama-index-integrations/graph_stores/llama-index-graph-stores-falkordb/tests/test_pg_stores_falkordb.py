from typing import Iterator

import pytest

from llama_index.core.graph_stores.types import (
    ChunkNode,
    EntityNode,
    PropertyGraphStore,
    Relation,
)
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.graph_stores.falkordb import (
    FalkorDBPGStore,
    FalkorDBPropertyGraphStore,
)
from llama_index.graph_stores.falkordb import falkordb_property_graph


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in FalkorDBPropertyGraphStore.__mro__]
    assert PropertyGraphStore.__name__ in names_of_base_classes
    assert FalkorDBPGStore is FalkorDBPropertyGraphStore


@pytest.fixture()
def pg_store(
    falkordb_url: str, graph_name: str
) -> Iterator[FalkorDBPropertyGraphStore]:
    store = FalkorDBPropertyGraphStore(url=falkordb_url, database=graph_name)
    store.structured_query("MATCH (n) DETACH DELETE n")
    yield store
    store.close()


def test_upsert_nodes_and_get(pg_store: FalkorDBPropertyGraphStore) -> None:
    entity = EntityNode(label="PERSON", name="Alice", properties={"age": 30})
    chunk = ChunkNode(text="Alice is a software engineer.")
    pg_store.upsert_nodes([entity, chunk])

    retrieved_entities = pg_store.get(ids=[entity.id])
    assert len(retrieved_entities) == 1
    assert isinstance(retrieved_entities[0], EntityNode)
    assert retrieved_entities[0].name == "Alice"
    assert retrieved_entities[0].label == "PERSON"

    retrieved_chunks = pg_store.get(ids=[chunk.id])
    assert len(retrieved_chunks) == 1
    assert isinstance(retrieved_chunks[0], ChunkNode)
    assert retrieved_chunks[0].text == "Alice is a software engineer."

    retrieved_by_prop = pg_store.get(properties={"age": 30})
    assert [node.id for node in retrieved_by_prop] == [entity.id]

    assert pg_store.get(properties={"non_existent_prop": "foo"}) == []


def test_get_with_ids_and_properties(pg_store: FalkorDBPropertyGraphStore) -> None:
    """Regression: combining `ids` and `properties` produced invalid Cypher."""
    alice = EntityNode(label="PERSON", name="Alice", properties={"age": 30})
    bob = EntityNode(label="PERSON", name="Bob", properties={"age": 40})
    pg_store.upsert_nodes([alice, bob])

    assert [n.name for n in pg_store.get(ids=[alice.id], properties={"age": 30})] == [
        "Alice"
    ]
    assert pg_store.get(ids=[alice.id], properties={"age": 40}) == []


def test_get_triplets_with_entity_names_and_ids(
    pg_store: FalkorDBPropertyGraphStore,
) -> None:
    """Regression: combining `entity_names` and `ids` produced invalid Cypher."""
    alice = EntityNode(label="PERSON", name="Alice")
    acme = EntityNode(label="ORGANIZATION", name="Acme")
    pg_store.upsert_nodes([alice, acme])
    pg_store.upsert_relations(
        [Relation(label="WORKS_FOR", source_id=alice.id, target_id=acme.id)]
    )

    triplets = pg_store.get_triplets(entity_names=["Alice"], ids=[alice.id])
    assert [t[1].label for t in triplets] == ["WORKS_FOR"]


def test_upsert_relations_is_idempotent(pg_store: FalkorDBPropertyGraphStore) -> None:
    """Regression: relations were CREATEd, duplicating on every re-ingest."""
    alice = EntityNode(label="PERSON", name="Alice")
    acme = EntityNode(label="ORGANIZATION", name="Acme")
    pg_store.upsert_nodes([alice, acme])
    relation = Relation(
        label="WORKS_FOR",
        source_id=alice.id,
        target_id=acme.id,
        properties={"since": 2023},
    )

    for _ in range(3):
        pg_store.upsert_relations([relation])

    count = pg_store.structured_query(
        "MATCH ()-[r:WORKS_FOR]->() RETURN count(r) AS count"
    )
    assert count[0]["count"] == 1


def test_labels_with_special_characters(pg_store: FalkorDBPropertyGraphStore) -> None:
    """Regression: labels were interpolated unquoted, breaking on spaces."""
    node = EntityNode(label="Business Unit", name="Sales")
    pg_store.upsert_nodes([node])

    assert [n.label for n in pg_store.get(ids=[node.id])] == ["Business Unit"]


def test_label_injection_is_neutralised(pg_store: FalkorDBPropertyGraphStore) -> None:
    """A backtick in an (LLM generated) label must not escape the identifier."""
    node = EntityNode(label="A` REMOVE e:`__Entity__", name="Injected")
    pg_store.upsert_nodes([node])

    labels = pg_store.structured_query(
        "MATCH (n) WHERE n.id = 'Injected' RETURN labels(n) AS labels"
    )[0]["labels"]
    assert "__Entity__" in labels


def test_relation_labels_with_special_characters(
    pg_store: FalkorDBPropertyGraphStore,
) -> None:
    alice = EntityNode(label="PERSON", name="Alice")
    acme = EntityNode(label="ORGANIZATION", name="Acme")
    pg_store.upsert_nodes([alice, acme])
    pg_store.upsert_relations(
        [Relation(label="WORKS FOR", source_id=alice.id, target_id=acme.id)]
    )

    triplets = pg_store.get_triplets(entity_names=["Alice"])
    assert [t[1].label for t in triplets] == ["WORKS FOR"]


def test_get_schema_str(pg_store: FalkorDBPropertyGraphStore) -> None:
    """Regression: `get_schema_str` raised TypeError on a non-empty graph."""
    alice = EntityNode(label="PERSON", name="Alice", properties={"age": 30})
    acme = EntityNode(label="ORGANIZATION", name="Acme")
    pg_store.upsert_nodes([alice, acme])
    pg_store.upsert_relations(
        [Relation(label="WORKS_FOR", source_id=alice.id, target_id=acme.id)]
    )

    schema_str = pg_store.get_schema_str(refresh=True)
    assert "PERSON {" in schema_str
    assert "age: INTEGER" in schema_str
    assert "WORKS_FOR" in schema_str
    # embeddings must never be advertised to the LLM
    assert "embedding" not in schema_str

    node_props = pg_store.get_schema()["node_props"]["PERSON"]
    assert {"property": "age", "type": "INTEGER"} in node_props


def test_vector_query_returns_similarity(pg_store: FalkorDBPropertyGraphStore) -> None:
    """Regression: a distance was returned, inverting the retriever ranking."""
    match = EntityNode(label="PERSON", name="Match", embedding=[1.0, 0.0, 0.0])
    other = EntityNode(label="PERSON", name="Other", embedding=[0.0, 1.0, 0.0])
    pg_store.upsert_nodes([match, other])

    nodes, scores = pg_store.vector_query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=2)
    )

    assert [node.name for node in nodes] == ["Match", "Other"]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0, abs=1e-4)


def test_vector_index_is_created(pg_store: FalkorDBPropertyGraphStore) -> None:
    pg_store.upsert_nodes(
        [EntityNode(label="PERSON", name="Alice", embedding=[1.0, 0.0, 0.0])]
    )
    assert pg_store._vector_index_dimensions == {"__Entity__": 3}


def test_vector_query_with_filters(pg_store: FalkorDBPropertyGraphStore) -> None:
    """Regression: filter values were interpolated instead of parameterised."""
    alice = EntityNode(label="PERSON", name="Alice", embedding=[1.0, 0.0, 0.0])
    bob = EntityNode(label="PERSON", name="Bob", embedding=[0.9, 0.1, 0.0])
    pg_store.upsert_nodes([alice, bob])

    nodes, _ = pg_store.vector_query(
        VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0],
            similarity_top_k=2,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="name", value="Bob", operator=FilterOperator.EQ)
                ]
            ),
        )
    )
    assert [node.name for node in nodes] == ["Bob"]


def test_vector_query_filter_value_is_not_executed(
    pg_store: FalkorDBPropertyGraphStore,
) -> None:
    """A malicious filter value must be treated as data, not as Cypher."""
    pg_store.upsert_nodes(
        [EntityNode(label="PERSON", name="Alice", embedding=[1.0, 0.0, 0.0])]
    )

    nodes, _ = pg_store.vector_query(
        VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0],
            similarity_top_k=2,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="name",
                        value="' OR true RETURN 1 //",
                        operator=FilterOperator.EQ,
                    )
                ]
            ),
        )
    )
    assert nodes == []
    assert (
        pg_store.structured_query("MATCH (n) RETURN count(n) AS count")[0]["count"] == 1
    )


def test_structured_query_returns_plain_values(
    pg_store: FalkorDBPropertyGraphStore,
) -> None:
    """Regression: raw driver objects leaked into the LLM context."""
    pg_store.upsert_nodes([EntityNode(label="PERSON", name="Alice")])

    row = pg_store.structured_query("MATCH (n) RETURN n")[0]["n"]
    assert isinstance(row, dict)
    assert row["name"] == "Alice"

    path = pg_store.structured_query("MATCH (n) RETURN [n] AS nodes")[0]["nodes"]
    assert isinstance(path, list)
    assert isinstance(path[0], dict)


def test_structured_query_sanitizes_embeddings(
    pg_store: FalkorDBPropertyGraphStore,
) -> None:
    """Regression: `sanitize_query_output` never applied to node objects."""
    pg_store.upsert_nodes(
        [EntityNode(label="PERSON", name="Alice", embedding=[0.1] * 256)]
    )

    row = pg_store.structured_query("MATCH (n) RETURN n")[0]["n"]
    assert "embedding" not in row


def test_range_indexes_are_created(pg_store: FalkorDBPropertyGraphStore) -> None:
    labels = {
        index["label"] for index in pg_store.structured_query("CALL db.indexes()")
    }
    assert {"__Entity__", "Chunk"} <= labels


def test_create_indexes_can_be_disabled(falkordb_url: str, graph_name: str) -> None:
    store = FalkorDBPropertyGraphStore(
        url=falkordb_url, database=graph_name, create_indexes=False
    )
    try:
        assert store.structured_query("CALL db.indexes()") == []
    finally:
        store.close()


def test_upsert_preserves_existing_embedding(
    pg_store: FalkorDBPropertyGraphStore,
) -> None:
    pg_store.upsert_nodes(
        [EntityNode(label="PERSON", name="Alice", embedding=[1.0, 0.0, 0.0])]
    )
    pg_store.upsert_nodes(
        [EntityNode(label="PERSON", name="Alice", properties={"age": 30})]
    )

    has_embedding = pg_store.structured_query(
        "MATCH (n) WHERE n.id = 'Alice' RETURN n.embedding IS NOT NULL AS has_embedding"
    )[0]["has_embedding"]
    assert has_embedding


def test_get_rel_map(pg_store: FalkorDBPropertyGraphStore) -> None:
    alice = EntityNode(label="PERSON", name="Alice")
    acme = EntityNode(label="ORGANIZATION", name="Acme")
    pg_store.upsert_nodes([alice, acme])
    pg_store.upsert_relations(
        [
            Relation(
                label="WORKS_FOR",
                source_id=alice.id,
                target_id=acme.id,
                properties={"since": 2023},
            )
        ]
    )

    paths = pg_store.get_rel_map(pg_store.get(ids=[alice.id]), depth=1)
    assert len(paths) == 1
    source, relation, target = paths[0]
    assert source.id == alice.id
    assert target.id == acme.id
    # relationship properties used to be dropped
    assert relation.properties == {"since": 2023}

    assert (
        pg_store.get_rel_map(
            pg_store.get(ids=[alice.id]), depth=1, ignore_rels=["WORKS_FOR"]
        )
        == []
    )


def test_llama_nodes_round_trip(pg_store: FalkorDBPropertyGraphStore) -> None:
    source_node = TextNode(text="Alice works for Acme since 2023.")
    entity = EntityNode(label="PERSON", name="Alice")
    pg_store.upsert_llama_nodes([source_node])
    pg_store.upsert_nodes([entity])
    pg_store.upsert_relations(
        [Relation(label="MENTIONS", source_id=source_node.node_id, target_id=entity.id)]
    )

    llama_nodes = pg_store.get_llama_nodes([source_node.node_id])
    assert [node.text for node in llama_nodes] == [source_node.text]

    pg_store.delete_llama_nodes(node_ids=[source_node.node_id])
    assert pg_store.get_llama_nodes([source_node.node_id]) == []


def test_mentions_relationship_from_triplet_source_id(
    pg_store: FalkorDBPropertyGraphStore,
) -> None:
    pg_store.upsert_nodes([ChunkNode(id_="chunk_1", text="Test chunk with entities")])
    pg_store.upsert_nodes(
        [
            EntityNode(
                label="PERSON",
                name="TestEntity",
                properties={"triplet_source_id": "chunk_1"},
            )
        ]
    )

    result = pg_store.structured_query(
        """
        MATCH (c:Chunk {id: 'chunk_1'})-[r:MENTIONS]->(e:__Entity__ {name: 'TestEntity'})
        RETURN count(r) AS mention_count
        """
    )
    assert result[0]["mention_count"] == 1


def test_delete(pg_store: FalkorDBPropertyGraphStore) -> None:
    alice = EntityNode(label="PERSON", name="Alice", properties={"age": 30})
    acme = EntityNode(label="ORGANIZATION", name="Acme")
    pg_store.upsert_nodes([alice, acme])
    pg_store.upsert_relations(
        [Relation(label="WORKS_FOR", source_id=alice.id, target_id=acme.id)]
    )

    pg_store.delete(relation_names=["WORKS_FOR"])
    assert pg_store.get_triplets(entity_names=["Alice"]) == []

    pg_store.delete(properties={"age": 30})
    assert pg_store.get(ids=[alice.id]) == []

    pg_store.delete(entity_names=["Acme"])
    assert pg_store.get(ids=[acme.id]) == []


def test_switch_graph(pg_store: FalkorDBPropertyGraphStore, graph_name: str) -> None:
    pg_store.upsert_nodes([EntityNode(label="PERSON", name="Alice")])
    pg_store.switch_graph(f"{graph_name}_other")

    assert pg_store.get(ids=["Alice"]) == []
    labels = {
        index["label"] for index in pg_store.structured_query("CALL db.indexes()")
    }
    assert "__Entity__" in labels


def test_context_manager_closes_connection(falkordb_url: str, graph_name: str) -> None:
    with FalkorDBPropertyGraphStore(url=falkordb_url, database=graph_name) as store:
        store.upsert_nodes([EntityNode(label="PERSON", name="Alice")])
    assert not hasattr(store, "_driver")


def test_client_kwargs_are_forwarded(falkordb_url: str, graph_name: str) -> None:
    store = FalkorDBPropertyGraphStore(
        url=falkordb_url, database=graph_name, socket_timeout=30
    )
    try:
        assert store.structured_query("RETURN 1 AS one")[0]["one"] == 1
    finally:
        store.close()


def test_batched_upserts(
    pg_store: FalkorDBPropertyGraphStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force several batches per label so the batching loops (and the per-label
    # grouping inside them) are actually exercised.
    monkeypatch.setattr(falkordb_property_graph, "CHUNK_SIZE", 10)

    entities = [
        EntityNode(
            label="PERSON" if i % 2 else "ORG",
            name=f"person_{i}",
            properties={"triplet_source_id": f"chunk_{i}"},
        )
        for i in range(105)
    ]
    relations = [
        Relation(label="KNOWS", source_id=f"person_{i}", target_id=f"person_{i + 1}")
        for i in range(104)
    ]
    pg_store.upsert_nodes(entities)
    pg_store.upsert_relations(relations)

    counts = pg_store.structured_query(
        "MATCH (n:`__Entity__`) RETURN count(n) AS nodes"
    )
    assert counts[0]["nodes"] == 105
    for label, expected in (("PERSON", 52), ("ORG", 53)):
        rows = pg_store.structured_query(
            f"MATCH (n:`{label}`) RETURN count(n) AS nodes"
        )
        assert rows[0]["nodes"] == expected
    rels = pg_store.structured_query("MATCH ()-[r:KNOWS]->() RETURN count(r) AS rels")
    assert rels[0]["rels"] == 104
    # MENTIONS is batched separately and must survive the batch boundary too.
    mentions = pg_store.structured_query(
        "MATCH (:Chunk)-[r:MENTIONS]->(:`__Entity__`) RETURN count(r) AS rels"
    )
    assert mentions[0]["rels"] == 105


def test_upsert_relations_before_nodes_does_not_duplicate(
    pg_store: FalkorDBPropertyGraphStore,
) -> None:
    """Relation endpoints must be reused by a later upsert_nodes, not duplicated."""
    pg_store.upsert_relations(
        [Relation(label="WORKS_FOR", source_id="Alice", target_id="Acme")]
    )
    pg_store.upsert_nodes(
        [
            EntityNode(label="PERSON", name="Alice", properties={"age": 30}),
            EntityNode(label="ORG", name="Acme"),
        ]
    )

    counts = pg_store.structured_query(
        "MATCH (n {id: 'Alice'}) RETURN count(n) AS nodes"
    )
    assert counts[0]["nodes"] == 1

    nodes = pg_store.get(ids=["Alice"])
    assert len(nodes) == 1
    assert nodes[0].properties["age"] == 30

    # The relation must still hang off that same node.
    triplets = pg_store.get_triplets(entity_names=["Alice"])
    assert len(triplets) == 1
    assert triplets[0][1].label == "WORKS_FOR"


def test_upsert_relations_uses_indexed_merge(
    pg_store: FalkorDBPropertyGraphStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MERGE must be label-scoped so FalkorDB can use the `id` index."""
    pg_store.upsert_nodes([EntityNode(label="PERSON", name="Alice")])

    captured = []
    original = pg_store.structured_query

    def spy(query, param_map=None):
        captured.append((query, param_map))
        return original(query, param_map)

    monkeypatch.setattr(pg_store, "structured_query", spy)
    pg_store.upsert_relations(
        [Relation(label="KNOWS", source_id="Alice", target_id="Bob")]
    )

    merges = [(q, p) for q, p in captured if "MERGE (source" in q]
    assert merges, "upsert_relations issued no MERGE statement"
    for query, param_map in merges:
        plan = str(pg_store._graph.explain(query, params=param_map))
        assert "All Node Scan" not in plan, plan


def test_get_triplets_returns_correct_properties_for_incoming_rels(
    pg_store: FalkorDBPropertyGraphStore,
) -> None:
    """Traversing an incoming edge must not swap the two nodes' property bags."""
    pg_store.upsert_nodes(
        [
            EntityNode(label="PERSON", name="Alice", properties={"city": "NYC"}),
            EntityNode(label="ORG", name="Acme", properties={"industry": "tech"}),
        ]
    )
    pg_store.upsert_relations(
        [
            Relation(
                label="WORKS_FOR",
                source_id="Alice",
                target_id="Acme",
                properties={"since": 2020},
            )
        ]
    )

    # Anchored on the *target*, so the store has to walk the edge backwards.
    triplets = pg_store.get_triplets(entity_names=["Acme"])
    assert len(triplets) == 1
    source, relation, target = triplets[0]

    assert source.name == "Alice"
    assert source.properties["city"] == "NYC"
    assert source.properties["id"] == "Alice"
    assert target.name == "Acme"
    assert target.properties["industry"] == "tech"
    assert target.properties["id"] == "Acme"
    assert relation.properties["since"] == 2020

    # The forward direction must agree with the reverse one.
    forward = pg_store.get_triplets(entity_names=["Alice"])
    assert [n.properties for n in (forward[0][0], forward[0][2])] == [
        source.properties,
        target.properties,
    ]


def test_refresh_schema_includes_rare_labels(
    pg_store: FalkorDBPropertyGraphStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label outside the sampling window must still appear in the schema."""
    monkeypatch.setattr(falkordb_property_graph, "SCHEMA_SAMPLE_SIZE", 5)

    pg_store.upsert_nodes(
        [EntityNode(label="BULK", name=f"bulk_{i}") for i in range(50)]
        + [EntityNode(label="RARE", name="rare_1", properties={"rare_prop": "x"})]
    )

    pg_store.refresh_schema()
    node_props = pg_store.structured_schema["node_props"]
    assert {"BULK", "RARE"} <= set(node_props)

    rare_props = {prop["property"] for prop in node_props["RARE"]}
    assert "rare_prop" in rare_props
    assert "embedding" not in rare_props

    # ...and the rendered schema the LLM sees must mention it too.
    assert "RARE" in pg_store.get_schema_str()


def _capture_queries(
    store: FalkorDBPropertyGraphStore, monkeypatch: pytest.MonkeyPatch
) -> list:
    """Record every Cypher statement a store method issues."""
    captured: list = []
    original = store.structured_query

    def spy(query, param_map=None):
        captured.append((query, param_map))
        return original(query, param_map)

    monkeypatch.setattr(store, "structured_query", spy)
    return captured


def _plans(store: FalkorDBPropertyGraphStore, captured: list) -> list:
    plans = []
    for query, params in captured:
        try:
            plans.append((query, str(store._graph.explain(query, params=params))))
        except Exception:
            # Procedure calls such as `CALL db.labels()` cannot be explained.
            continue
    return plans


def _seed_graph(store: FalkorDBPropertyGraphStore) -> None:
    store.upsert_nodes(
        [
            EntityNode(label="PERSON", name="Alice", properties={"city": "NYC"}),
            EntityNode(label="ORG", name="Acme"),
            ChunkNode(text="some chunk text", id_="chunk_1"),
        ]
    )
    store.upsert_relations(
        [Relation(label="WORKS_FOR", source_id="Alice", target_id="Acme")]
    )


def test_get_uses_indexes_instead_of_scanning_the_graph(
    pg_store: FalkorDBPropertyGraphStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`get()` runs twice per ingestion batch to dedup - it must not full-scan."""
    _seed_graph(pg_store)

    captured = _capture_queries(pg_store, monkeypatch)
    pg_store.get(ids=["Alice", "chunk_1"])

    plans = _plans(pg_store, captured)
    assert plans, "get() issued no explainable query"
    for query, plan in plans:
        assert "All Node Scan" not in plan, f"{query}\n{plan}"
        assert "Node By Index Scan" in plan, f"{query}\n{plan}"


def test_get_still_returns_entities_and_chunks(
    pg_store: FalkorDBPropertyGraphStore,
) -> None:
    """Splitting the lookup per label must not change what `get()` returns."""
    _seed_graph(pg_store)

    by_id = {node.id: node for node in pg_store.get(ids=["Alice", "chunk_1"])}
    assert set(by_id) == {"Alice", "chunk_1"}
    assert isinstance(by_id["Alice"], EntityNode)
    assert by_id["Alice"].properties["city"] == "NYC"
    assert isinstance(by_id["chunk_1"], ChunkNode)
    assert by_id["chunk_1"].text == "some chunk text"

    # No filters must still return every node the store manages, exactly once.
    all_ids = [node.id for node in pg_store.get()]
    assert sorted(all_ids) == ["Acme", "Alice", "chunk_1"]

    # Property filters keep working across both labels.
    assert [n.id for n in pg_store.get(properties={"city": "NYC"})] == ["Alice"]


def test_get_deduplicates_nodes_carrying_both_labels(
    pg_store: FalkorDBPropertyGraphStore,
) -> None:
    _seed_graph(pg_store)
    pg_store.structured_query("MATCH (n {id: 'Alice'}) SET n:Chunk")

    assert [node.id for node in pg_store.get(ids=["Alice"])] == ["Alice"]


def test_refresh_schema_does_not_traverse_every_relationship(
    pg_store: FalkorDBPropertyGraphStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PropertyGraphIndex refreshes the schema after every ingestion batch."""
    _seed_graph(pg_store)

    captured = _capture_queries(pg_store, monkeypatch)
    pg_store.refresh_schema()

    traversals = [
        (query, plan)
        for query, plan in _plans(pg_store, captured)
        if "Conditional Traverse" in plan
    ]
    assert traversals, "expected refresh_schema to inspect relationships"
    for query, plan in traversals:
        assert "Limit" in plan, f"unbounded relationship traversal:\n{query}\n{plan}"


def test_refresh_schema_still_reports_relationships(
    pg_store: FalkorDBPropertyGraphStore,
) -> None:
    _seed_graph(pg_store)
    pg_store.refresh_schema()

    triples = {
        (rel["start"], rel["type"], rel["end"])
        for rel in pg_store.structured_schema["relationships"]
    }
    assert ("PERSON", "WORKS_FOR", "ORG") in triples
    assert ("Chunk", "MENTIONS", "PERSON") not in triples  # no source id seeded
    assert "(:PERSON)-[:WORKS_FOR]->(:ORG)" in pg_store.get_schema_str()


def test_delete_uses_indexes_instead_of_scanning_the_graph(
    pg_store: FalkorDBPropertyGraphStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_graph(pg_store)

    captured = _capture_queries(pg_store, monkeypatch)
    pg_store.delete(ids=["chunk_1"])

    plans = _plans(pg_store, captured)
    assert plans, "delete() issued no explainable query"
    for query, plan in plans:
        assert "All Node Scan" not in plan, f"{query}\n{plan}"

    assert [n.id for n in pg_store.get(ids=["chunk_1"])] == []
    assert sorted(n.id for n in pg_store.get()) == ["Acme", "Alice"]


def test_delete_by_entity_names_and_properties_still_works(
    pg_store: FalkorDBPropertyGraphStore,
) -> None:
    _seed_graph(pg_store)

    pg_store.delete(entity_names=["Acme"])
    assert sorted(n.id for n in pg_store.get()) == ["Alice", "chunk_1"]

    pg_store.delete(properties={"city": "NYC"})
    assert sorted(n.id for n in pg_store.get()) == ["chunk_1"]
