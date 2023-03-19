"""Test pandas index."""

from typing import Any, Dict, Tuple, cast

import pandas as pd
import pytest

from gpt_index.indices.struct_store.pandas import GPTPandasIndex
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_prompts import MOCK_PANDAS_PROMPT


@pytest.fixture
def struct_kwargs() -> Tuple[Dict, Dict]:
    """Index kwargs."""
    # NOTE: QuestionAnswer and Refine templates aren't technically used
    index_kwargs: Dict[str, Any] = {}
    query_kwargs = {
        "pandas_prompt": MOCK_PANDAS_PROMPT,
    }
    return index_kwargs, query_kwargs


@patch_common
def test_pandas_index(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Tuple[Dict, Dict],
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
    response = index.query("population", verbose=True)
    import sys

    if sys.version_info < (3, 9):
        assert response.response == 'df["population"].iloc[0]'
    else:
        assert response.response == "2930000"
    extra_info = cast(Dict[str, Any], response.extra_info)
    assert extra_info["pandas_instruction_str"] == ('df["population"].iloc[0]')
