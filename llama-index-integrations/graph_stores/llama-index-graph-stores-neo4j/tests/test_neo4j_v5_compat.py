import inspect

import llama_index.graph_stores.neo4j.neo4j_property_graph as neo4j_pg

import pytest
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


def test_no_legacy_call_syntax():
    """Ensure no deprecated Neo4j 4.x CALL (vars) syntax exists."""
    source = inspect.getsource(neo4j_pg)

    assert "CALL (" not in source, (
        "Legacy CALL (vars) syntax detected. Neo4j 5 requires CALL { WITH vars ... }"
    )


# this test is primarily to ensure that the embedding subquery executes without CypherSyntaxError.
# It assumes a local Neo4j instance is running with the specified credentials.
@pytest.mark.integration
def test_node_insert_with_embedding():
    """Ensure embedding subquery executes without CypherSyntaxError."""
    store = Neo4jPropertyGraphStore(
        url="bolt://localhost:7687",
        username="neo4j",
        password="admin_password",
    )

    test_data = [
        {
            "id": "test-node",
            "label": "TestLabel",
            "name": "Test Node",
            "properties": {"triplet_source_id": None},
            "embedding": [0.1] * 384,
        }
    ]

    # Should not raise CypherSyntaxError
    store.upsert_nodes(test_data)

    # Cleanup
    store.structured_query("MATCH (n {id:'test-node'}) DETACH DELETE n")
