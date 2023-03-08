"""Test empty index."""

from typing import Any, Dict, Tuple

import pytest

from gpt_index.data_structs.data_structs import EmptyIndex
from gpt_index.indices.empty.base import GPTEmptyIndex
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_prompts import MOCK_INPUT_PROMPT


@pytest.fixture
def struct_kwargs() -> Tuple[Dict, Dict]:
    """Index kwargs."""
    index_kwargs: Dict[str, Any] = {}
    query_kwargs = {
        "input_prompt": MOCK_INPUT_PROMPT,
    }
    return index_kwargs, query_kwargs


@patch_common
def test_empty(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Tuple[Dict, Dict],
) -> None:
    """Test build list."""
    index_kwargs, query_kwargs = struct_kwargs
    empty_index = GPTEmptyIndex()
    assert isinstance(empty_index.index_struct, EmptyIndex)

    response = empty_index.query("What is?", **query_kwargs)
    assert response.response == "What is?"
    assert response.source_nodes == []
