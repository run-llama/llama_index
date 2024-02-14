from unittest.mock import MagicMock, patch

from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore


@patch("llama_index.graph_stores.neo4j.Neo4jGraphStore")
def test_kuzu_graph_store(MockNeo4jGraphStore: MagicMock):
    instance: Neo4jGraphStore = MockNeo4jGraphStore.return_value()
    assert isinstance(instance, GraphStore)
