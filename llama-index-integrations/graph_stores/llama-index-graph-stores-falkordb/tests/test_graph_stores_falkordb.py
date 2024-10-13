from unittest.mock import MagicMock, patch

from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.falkordb.base import FalkorDBGraphStore


@patch("llama_index.graph_stores.falkordb.base.FalkorDBGraphStore")
def test_falkordb_class(MockFalkorDBGraphStore: MagicMock):
    instance: FalkorDBGraphStore = MockFalkorDBGraphStore.return_value()
    assert isinstance(instance, GraphStore)


@patch("llama_index.graph_stores.falkordb.base.FalkorDBGraphStore")
def test_falkordb_store_data(MockFalkorDBGraphStore: MagicMock):
    instance: FalkorDBGraphStore = MockFalkorDBGraphStore.return_value()
    data = {"key": "value"}
    instance.store_data(data)
    instance.store_data.assert_called_once_with(data)


@patch("llama_index.graph_stores.falkordb.base.FalkorDBGraphStore")
def test_falkordb_retrieve_data(MockFalkorDBGraphStore: MagicMock):
    instance: FalkorDBGraphStore = MockFalkorDBGraphStore.return_value()
    key = "key"
    instance.retrieve_data(key)
    instance.retrieve_data.assert_called_once_with(key)


@patch("llama_index.graph_stores.falkordb.base.FalkorDBGraphStore")
def test_falkordb_delete_data(MockFalkorDBGraphStore: MagicMock):
    instance: FalkorDBGraphStore = MockFalkorDBGraphStore.return_value()
    key = "key"
    instance.delete_data(key)
    instance.delete_data.assert_called_once_with(key)
