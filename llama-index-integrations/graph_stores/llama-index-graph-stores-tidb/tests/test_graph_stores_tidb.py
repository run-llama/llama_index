from unittest.mock import MagicMock, patch

from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.tidb import TiDBGraphStore


@patch("llama_index.graph_stores.tidb.TiDBGraphStore")
def test_tidb_graph_store(MockTiDBGraphStore: MagicMock):
    instance: TiDBGraphStore = MockTiDBGraphStore.return_value()
    assert isinstance(instance, GraphStore)
