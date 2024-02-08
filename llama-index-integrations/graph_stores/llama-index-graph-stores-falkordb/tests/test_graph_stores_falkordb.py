from unittest.mock import patch

from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.falkordb.base import FalkorDBGraphStore


@patch("llama_index.graph_stores.falkordb.base.FalkorDBGraphStore")
def test_falkordb_class(mock_db: FalkorDBGraphStore):
    assert isinstance(mock_db, GraphStore)
