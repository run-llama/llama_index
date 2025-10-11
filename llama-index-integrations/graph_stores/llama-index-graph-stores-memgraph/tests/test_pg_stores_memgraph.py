import os
import pytest
from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
)
from llama_index.core.schema import TextNode

memgraph_user = os.environ.get("MEMGRAPH_TEST_USER")
memgraph_pass = os.environ.get("MEMGRAPH_TEST_PASS")
memgraph_url = os.environ.get("MEMGRAPH_TEST_URL")

if not memgraph_user or not memgraph_pass or not memgraph_url:
    MEMGRAPH_AVAILABLE = False
else:
    MEMGRAPH_AVAILABLE = True


@pytest.fixture()
def pg_store() -> MemgraphPropertyGraphStore:
    """Define Memgraph graph store."""
    if not MEMGRAPH_AVAILABLE:
        pytest.skip("No Memgraph credentials provided")
    return MemgraphPropertyGraphStore(
        username=memgraph_user, password=memgraph_pass, url=memgraph_url
    )


def test_memgraph_pg_store(pg_store: MemgraphPropertyGraphStore) -> None:
    """Test functions for Memgraph graph store."""
    # Clear the database
    pg_store.structured_query("STORAGE MODE IN_MEMORY_ANALYTICAL")
    pg_store.structured_query("DROP GRAPH")
    pg_store.structured_query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")

    # Test inserting nodes into Memgraph.
    entity1 = EntityNode(label="PERSON", name="Logan", properties={"age": 28})
    entity2 = EntityNode(label="ORGANIZATION", name="LlamaIndex")
    pg_store.upsert_nodes([entity1, entity2])
    # Assert the nodes are inserted correctly
    kg_nodes = pg_store.get(ids=[entity1.id])
    assert kg_nodes[0].name == entity1.name

    # Test inserting relations into Memgraph.
    relation = Relation(
        label="WORKS_FOR",
        source_id=entity1.id,
        target_id=entity2.id,
        properties={"since": 2023},
    )

    pg_store.upsert_relations([relation])
    # Assert the relation is inserted correctly by retrieving the relation map
    kg_nodes = pg_store.get(ids=[entity1.id])
    pg_store.get_rel_map(kg_nodes, depth=1)

    # Test inserting a source text node and 'MENTIONS' relations.
    source_node = TextNode(text="Logan (age 28), works for 'LlamaIndex' since 2023.")

    relations = [
        Relation(label="MENTIONS", target_id=entity1.id, source_id=source_node.node_id),
        Relation(label="MENTIONS", target_id=entity2.id, source_id=source_node.node_id),
    ]

    pg_store.upsert_llama_nodes([source_node])
    pg_store.upsert_relations(relations)

    # Assert the source node and relations are inserted correctly
    pg_store.get_llama_nodes([source_node.node_id])

    # Test retrieving nodes by properties.
    kg_nodes = pg_store.get(properties={"age": 28})

    # Test executing a structured query in Memgraph.
    query = "MATCH (n:`__Entity__`) RETURN n"
    pg_store.structured_query(query)

    # Test upserting a new node with additional properties.
    new_node = EntityNode(
        label="PERSON", name="Logan", properties={"age": 28, "location": "Canada"}
    )
    pg_store.upsert_nodes([new_node])

    # Assert the node has been updated with the new property
    kg_nodes = pg_store.get(properties={"age": 28})

    # Test deleting nodes from Memgraph.
    pg_store.delete(ids=[source_node.node_id])
    pg_store.delete(ids=[entity1.id, entity2.id])

    # Assert the nodes have been deleted
    pg_store.get(ids=[entity1.id, entity2.id])
