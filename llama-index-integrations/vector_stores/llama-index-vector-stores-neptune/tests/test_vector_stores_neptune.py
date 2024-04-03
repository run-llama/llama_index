from unittest.mock import MagicMock, patch

from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.neptune import NeptuneAnalyticsVectorStore


@patch("llama_index.vector_stores.neptune.NeptuneAnalyticsVectorStore")
def test_neptune_analytics_vector_store(MockNeptuneAnalyticsGraphStore: MagicMock):
    instance: NeptuneAnalyticsVectorStore = (
        MockNeptuneAnalyticsGraphStore.return_value()
    )
    assert isinstance(instance, VectorStore)
