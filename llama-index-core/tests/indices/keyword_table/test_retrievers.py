from typing import List
from unittest.mock import patch

from llama_index.core.indices.keyword_table.simple_base import (
    SimpleKeywordTableIndex,
)
from llama_index.core.schema import Document, QueryBundle
from tests.mock_utils.mock_utils import mock_extract_keywords


@patch(
    "llama_index.core.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
@patch(
    "llama_index.core.indices.keyword_table.retrievers.simple_extract_keywords",
    mock_extract_keywords,
)
def test_retrieve(
    documents: List[Document], mock_embed_model, patch_token_text_splitter
) -> None:
    """Test query."""
    # test simple keyword table
    # NOTE: here the keyword extraction isn't mocked because we're using
    # the regex-based keyword extractor, not GPT
    table = SimpleKeywordTableIndex.from_documents(documents)

    retriever = table.as_retriever(
        retriever_mode="simple", embed_model=mock_embed_model
    )
    nodes = retriever.retrieve(QueryBundle("Hello"))
    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "Hello world."
