import inspect
import re

import llama_index.graph_stores.neo4j.neo4j_property_graph as neo4j_pg

import pytest
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


def test_no_legacy_call_syntax():
    """Ensure no deprecated Neo4j 4.x CALL (vars) syntax exists."""
    source = inspect.getsource(neo4j_pg)

    assert "CALL (" not in source, (
        "Legacy CALL (vars) syntax detected. Neo4j 5 requires CALL { WITH vars ... }"
    )


def test_no_deprecated_cypher_patterns():
    """Guard against other Cypher patterns removed in Neo4j 5."""
    source = inspect.getsource(neo4j_pg)

    forbidden_patterns = {
        r"USING\s+PERIODIC\s+COMMIT": "USING PERIODIC COMMIT was removed; use CALL { ... } IN TRANSACTIONS",
        r"CALL\s+db\.index(?:es)?\b(?!\.vector)": "db.index* procedures (except vector) are deprecated; use SHOW INDEXES",
        r"CALL\s+db\.constraints\b": "db.constraints was removed; use SHOW CONSTRAINTS",
    }

    for pattern, reason in forbidden_patterns.items():
        assert not re.search(pattern, source, flags=re.IGNORECASE), (
            f"Deprecated Neo4j pattern detected: '{pattern}' ({reason})"
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
