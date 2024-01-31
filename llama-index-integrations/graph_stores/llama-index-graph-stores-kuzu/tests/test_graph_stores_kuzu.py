from llama_index.core.graph_stores.types import GraphStore
from unittest.mock import MagicMock, patch


@patch("llama_index.graph_stores.kuzu.KuzuGraphStore")
def test_kuzu_graph_store(MockKuzuGraphStore: MagicMock):
    instance = MockKuzuGraphStore.return_value()
    assert isinstance(instance, GraphStore)
