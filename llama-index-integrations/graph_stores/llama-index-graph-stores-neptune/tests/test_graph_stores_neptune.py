from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.neptune import (
    NeptuneAnalyticsGraphStore,
    NeptuneDatabaseGraphStore,
)


def test_neptune_analytics_graph_store():
    names_of_bases = [b.__name__ for b in NeptuneAnalyticsGraphStore.__bases__]
    assert GraphStore.__name__ in names_of_bases


def test_neptune_database_graph_store():
    names_of_bases = [b.__name__ for b in NeptuneDatabaseGraphStore.__bases__]
    assert GraphStore.__name__ in names_of_bases
