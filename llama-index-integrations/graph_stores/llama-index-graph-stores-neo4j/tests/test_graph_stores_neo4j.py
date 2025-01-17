from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore


def test_neo4j_graph_store():
    names_of_bases = [b.__name__ for b in Neo4jGraphStore.__bases__]
    assert GraphStore.__name__ in names_of_bases
