from unittest.mock import MagicMock, patch

from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.kuzu import KuzuGraphStore


@patch("llama_index.graph_stores.kuzu.KuzuGraphStore")
def test_kuzu_graph_store(MockKuzuGraphStore: MagicMock):
    instance: KuzuGraphStore = MockKuzuGraphStore.return_value()
    assert isinstance(instance, GraphStore)
