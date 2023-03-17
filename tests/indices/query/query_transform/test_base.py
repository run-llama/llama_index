"""Test query transform."""

from typing import Any, List

import pytest

from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.query.query_transform.base import DecomposeQueryTransform
from gpt_index.readers.schema.base import Document
from tests.indices.query.query_transform.mock_utils import MOCK_DECOMPOSE_PROMPT
from tests.mock_utils.mock_decorator import patch_common


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Correct.\n"
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(doc_text)]


@patch_common
def test_decompose_query_transform(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test decompose query transform."""
    query_transform = DecomposeQueryTransform(
        decompose_query_prompt=MOCK_DECOMPOSE_PROMPT
    )

    # initialize list index
    # documents aren't used for this test
    index = GPTListIndex(documents)
    index.set_text("Foo bar")
    query_str = "What is?"
    new_query_bundle = query_transform.run(
        query_str, {"index_struct": index.index_struct}
    )
    assert new_query_bundle.query_str == "What is?:Foo bar"
    assert new_query_bundle.embedding_strs == ["What is?:Foo bar"]
