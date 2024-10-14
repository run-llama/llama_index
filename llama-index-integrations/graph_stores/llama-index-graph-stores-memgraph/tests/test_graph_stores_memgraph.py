from unittest.mock import MagicMock, patch

from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.memgraph import MemgraphGraphStore


@patch("llama_index.graph_stores.memgraph.MemgraphGraphStore")
def test_memgraph_graph_store(MockMemgraphGraphStore: MagicMock):
    instance: MemgraphGraphStore = MockMemgraphGraphStore.return_value()
    assert isinstance(instance, GraphStore)
