from unittest.mock import MagicMock, patch

from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.falkordb.base import FalkorDBGraphStore


@patch("llama_index.graph_stores.falkordb.base.FalkorDBGraphStore")
def test_falkordb_class(MockFalkorDBGraphStore: MagicMock):
    instance: FalkorDBGraphStore = MockFalkorDBGraphStore.return_value()
    assert isinstance(instance, GraphStore)
