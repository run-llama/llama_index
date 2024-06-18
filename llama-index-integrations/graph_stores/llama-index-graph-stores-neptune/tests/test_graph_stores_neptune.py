from unittest.mock import MagicMock, patch

from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.neptune import (
    NeptuneAnalyticsGraphStore,
    NeptuneDatabaseGraphStore,
)


@patch("llama_index.graph_stores.neptune.NeptuneAnalyticsGraphStore")
def test_neptune_analytics_graph_store(MockNeptuneAnalyticsGraphStore: MagicMock):
    instance: NeptuneAnalyticsGraphStore = MockNeptuneAnalyticsGraphStore.return_value()
    assert isinstance(instance, GraphStore)


@patch("llama_index.graph_stores.neptune.NeptuneDatabaseGraphStore")
def test_neptune_analytics_graph_store(MockNeptuneDatabaseGraphStore: MagicMock):
    instance: NeptuneDatabaseGraphStore = MockNeptuneDatabaseGraphStore.return_value()
    assert isinstance(instance, GraphStore)
