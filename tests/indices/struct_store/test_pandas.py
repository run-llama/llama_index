"""Test pandas index."""

from typing import Any, Dict, cast

import pandas as pd
from gpt_index.indices.query.schema import QueryBundle

from gpt_index.indices.struct_store.pandas import GPTPandasIndex
from gpt_index.indices.struct_store.pandas_query import GPTNLPandasQueryEngine
from tests.mock_utils.mock_decorator import patch_common


@patch_common
def test_pandas_index(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
) -> None:
    """Test GPTPandasIndex."""
    # Test on some sample data
    df = pd.DataFrame(
        {
            "city": ["Toronto", "Tokyo", "Berlin"],
            "population": [2930000, 13960000, 3645000],
        }
    )
    index = GPTPandasIndex(df=df)
    # the mock prompt just takes the first item in the given column
    query_engine = GPTNLPandasQueryEngine(index=index, verbose=True)
    response = query_engine.query(QueryBundle("population"))
    import sys

    if sys.version_info < (3, 9):
        assert str(response) == 'df["population"].iloc[0]'
    else:
        assert str(response) == "2930000"
    extra_info = cast(Dict[str, Any], response.extra_info)
    assert extra_info["pandas_instruction_str"] == ('df["population"].iloc[0]')
