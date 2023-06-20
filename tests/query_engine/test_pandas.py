"""Test pandas index."""

from typing import Any, Dict, cast

import pandas as pd

from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.struct_store.pandas import PandasIndex
from llama_index.query_engine.pandas_query_engine import PandasQueryEngine


def test_pandas_query_engine(mock_service_context: ServiceContext) -> None:
    """Test PandasIndex."""
    # Test on some sample data
    df = pd.DataFrame(
        {
            "city": ["Toronto", "Tokyo", "Berlin"],
            "population": [2930000, 13960000, 3645000],
        }
    )
    # the mock prompt just takes the first item in the given column
    query_engine = PandasQueryEngine(
        df, service_context=mock_service_context, verbose=True
    )
    response = query_engine.query(QueryBundle("population"))
    import sys

    if sys.version_info < (3, 9):
        assert str(response) == 'df["population"].iloc[0]'
    else:
        assert str(response) == "2930000"
    extra_info = cast(Dict[str, Any], response.extra_info)
    assert extra_info["pandas_instruction_str"] == ('df["population"].iloc[0]')
