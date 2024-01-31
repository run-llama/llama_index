from llama_index.core.graph_stores.types import GraphStore
from unittest.mock import patch


@patch("llama_index.graph_stores.falkordb.base.FalkorDBGraphStore")
def test_falkordb_class(mock_db):
    assert isinstance(mock_db, GraphStore)
