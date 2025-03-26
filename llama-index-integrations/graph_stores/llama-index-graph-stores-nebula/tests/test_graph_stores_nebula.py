from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.nebula import NebulaGraphStore


def test_nebula_graph_store():
    names_of_bases = [b.__name__ for b in NebulaGraphStore.__bases__]
    assert GraphStore.__name__ in names_of_bases
