from unittest.mock import MagicMock, patch

from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.nebula import NebulaGraphStore


@patch("llama_index.graph_stores.nebula.NebulaGraphStore")
def test_kuzu_graph_store(MockNebulaGraphStore: MagicMock):
    instance: NebulaGraphStore = MockNebulaGraphStore.return_value()
    assert isinstance(instance, GraphStore)
