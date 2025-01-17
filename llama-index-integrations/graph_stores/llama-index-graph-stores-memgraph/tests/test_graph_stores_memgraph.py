from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.memgraph import MemgraphGraphStore


def test_memgraph_graph_store():
    names_of_bases = [b.__name__ for b in MemgraphGraphStore.__bases__]
    assert GraphStore.__name__ in names_of_bases
