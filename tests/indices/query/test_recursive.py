"""Test recursive queries."""

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from gpt_index.indices.data_structs import IndexGraph, Node
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.query.schema import QueryConfig
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.schema import Document
from tests.mock_utils.mock_predict import mock_openai_llm_predict
from tests.mock_utils.mock_prompts import (
    MOCK_INSERT_PROMPT,
    MOCK_QUERY_PROMPT,
    MOCK_REFINE_PROMPT,
    MOCK_SUMMARY_PROMPT,
    MOCK_TEXT_QA_PROMPT,
)
from tests.mock_utils.mock_text_splitter import mock_token_splitter_newline


@pytest.fixture
def struct_kwargs() -> Tuple[Dict, List]:
    """Index kwargs."""
    index_kwargs = {
        "tree": {
            "summary_template": MOCK_SUMMARY_PROMPT,
            "insert_prompt": MOCK_INSERT_PROMPT,
            "num_children": 2,
        },
        "list": {
            "text_qa_template": MOCK_TEXT_QA_PROMPT,
        },
    }
    query_configs = [
        QueryConfig(
            index_struct_type="tree",
            query_mode="default",
            query_kwargs={
                "query_template": MOCK_QUERY_PROMPT,
                "text_qa_template": MOCK_TEXT_QA_PROMPT,
                "refine_template": MOCK_REFINE_PROMPT,
            },
        ),
        QueryConfig(
            index_struct_type="list",
            query_mode="default",
            query_kwargs={
                "text_qa_template": MOCK_TEXT_QA_PROMPT,
                "refine_template": MOCK_REFINE_PROMPT,
            },
        ),
    ]
    return index_kwargs, query_configs


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    docs = [
        Document("This is a test v2."),
        Document("This is another test."),
        Document("This is a test."),
        Document("Hello world."),
        Document("Hello world."),
        Document("This is a test."),
        Document("This is another test."),
        Document("This is a test v2."),
    ]
    return docs


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "predict", side_effect=mock_openai_llm_predict)
@patch.object(LLMPredictor, "__init__", return_value=None)
def test_recursive_query_list_tree(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test query."""

    index_kwargs, query_configs = struct_kwargs
    list_kwargs = index_kwargs["list"]
    tree_kwargs = index_kwargs["tree"]
    # try building a list for every two, then a tree
    list1 = GPTListIndex(documents[0:2], **list_kwargs)
    list1.set_text("summary1")
    list2 = GPTListIndex(documents[2:4], **list_kwargs)
    list2.set_text("summary2")
    list3 = GPTListIndex(documents[4:6], **list_kwargs)
    list3.set_text("summary3")
    list4 = GPTListIndex(documents[6:8], **list_kwargs)
    list4.set_text("summary4")

    # there are two root nodes in this tree: one containing [list1, list2]
    # and the other containing [list3, list4]
    tree = GPTTreeIndex(
        [
            list1,
            list2,
            list3,
            list4,
        ],
        **tree_kwargs
    )
    query_str = "What is?"
    # query should first pick the left root node, then pick list1
    # within list1, it should go through the first document and second document
    response = tree.query(query_str, mode="recursive", query_configs=query_configs)
    assert response == ("What is?:This is a test v2.")

    # Also test a non-recursive query. This should not go down into the list
    tree_query_kwargs = query_configs[0].query_kwargs
    response = tree.query(query_str, mode="default", **tree_query_kwargs)
    assert response == ("What is?:summary1")
