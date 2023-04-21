from typing import Any, List
from unittest.mock import patch
from gpt_index.indices.keyword_table.simple_base import GPTSimpleKeywordTableIndex
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_utils import mock_extract_keywords


@patch_common
@patch(
    "gpt_index.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
@patch(
    "gpt_index.indices.keyword_table.retrievers.simple_extract_keywords",
    mock_extract_keywords,
)
def test_retrieve(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test query."""
    # test simple keyword table
    # NOTE: here the keyword extraction isn't mocked because we're using
    # the regex-based keyword extractor, not GPT
    table = GPTSimpleKeywordTableIndex.from_documents(documents)

    retriever = table.as_retriever(mode="simple")
    nodes = retriever.retrieve(QueryBundle("Hello"))
    assert len(nodes) == 1
    assert nodes[0].node.text == "Hello world."
