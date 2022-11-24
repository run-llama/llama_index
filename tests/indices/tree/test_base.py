"""Test tree index."""

from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.schema import Document
from tests.mock_utils.mock_predict import mock_openai_llm_predict
from tests.mock_utils.mock_prompts import (
    MOCK_QUERY_PROMPT,
    MOCK_REFINE_PROMPT,
    MOCK_SUMMARY_PROMPT,
    MOCK_TEXT_QA_PROMPT,
)
from tests.mock_utils.mock_text_splitter import mock_token_splitter_newline


@pytest.fixture
def struct_kwargs() -> Tuple[Dict, Dict]:
    """Index kwargs."""
    index_kwargs = {
        "summary_template": MOCK_SUMMARY_PROMPT,
        "num_children": 2,
    }
    query_kwargs = {
        "query_template": MOCK_QUERY_PROMPT,
        "text_qa_template": MOCK_TEXT_QA_PROMPT,
        "refine_template": MOCK_REFINE_PROMPT,
    }
    return index_kwargs, query_kwargs


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(doc_text)]


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch("gpt_index.indices.tree.base.openai_llm_predict", mock_openai_llm_predict)
def test_build_tree(
    _mock_predict: Any, documents: List[Document], struct_kwargs: Dict
) -> None:
    """Test build tree."""
    index_kwargs, _ = struct_kwargs
    tree = GPTTreeIndex(documents, **index_kwargs)
    assert len(tree.index_struct.all_nodes) == 6
    # check contents of nodes
    assert tree.index_struct.all_nodes[0].text == "Hello world."
    assert tree.index_struct.all_nodes[1].text == "This is a test."
    assert tree.index_struct.all_nodes[2].text == "This is another test."
    assert tree.index_struct.all_nodes[3].text == "This is a test v2."
    assert tree.index_struct.all_nodes[4].text == ("Hello world.\nThis is a test.")
    assert tree.index_struct.all_nodes[5].text == (
        "This is another test.\nThis is a test v2."
    )


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch("gpt_index.indices.tree.base.openai_llm_predict", mock_openai_llm_predict)
@patch("gpt_index.indices.tree.leaf_query.openai_llm_predict", mock_openai_llm_predict)
def test_query(
    _mock_predict: Any, documents: List[Document], struct_kwargs: Dict
) -> None:
    """Test query."""
    index_kwargs, query_kwargs = struct_kwargs
    tree = GPTTreeIndex(documents, **index_kwargs)

    # test default query
    query_str = "What is?"
    response = tree.query(query_str, mode="default", **query_kwargs)
    assert response == ("What is?\n" "Hello world.")
