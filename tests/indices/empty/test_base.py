"""Test empty index."""

from typing import Any

from gpt_index.data_structs.data_structs_v2 import EmptyIndex
from gpt_index.indices.empty.base import GPTEmptyIndex
from tests.mock_utils.mock_decorator import patch_common


@patch_common
def test_empty(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
) -> None:
    """Test build list."""
    empty_index = GPTEmptyIndex()
    assert isinstance(empty_index.index_struct, EmptyIndex)

    retriever = empty_index.as_retriever()
    nodes = retriever.retrieve("What is?")
    assert len(nodes) == 0
