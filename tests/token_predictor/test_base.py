"""Test token predictor."""

from typing import Any
from unittest.mock import patch

from gpt_index.indices.keyword_table.base import GPTKeywordTableIndex
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.readers.schema.base import Document
from gpt_index.token_counter.mock_chain_wrapper import MockLLMPredictor
from tests.mock_utils.mock_text_splitter import mock_token_splitter_newline


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
def test_token_predictor(mock_split: Any) -> None:
    """Test token predictor."""
    # here, just assert that token predictor runs (before checking behavior)
    # TODO: mock token counting a bit more carefully
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    document = Document(doc_text)
    llm_predictor = MockLLMPredictor(max_tokens=256)

    # test tree index
    index = GPTTreeIndex([document], llm_predictor=llm_predictor)
    index.query("What is?", llm_predictor=llm_predictor)

    # test keyword table index
    index_keyword = GPTKeywordTableIndex([document], llm_predictor=llm_predictor)
    index_keyword.query("What is?", llm_predictor=llm_predictor)

    # test list index
    index_list = GPTListIndex([document], llm_predictor=llm_predictor)
    index_list.query("What is?", llm_predictor=llm_predictor)
