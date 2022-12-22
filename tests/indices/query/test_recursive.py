"""Test recursive queries."""

from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

from gpt_index.indices.data_structs import IndexStructType
from gpt_index.indices.keyword_table.simple_base import GPTSimpleKeywordTableIndex
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.query.schema import QueryConfig, QueryMode
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMChain, LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_predict import (
    mock_llmchain_predict,
    mock_llmpredictor_predict,
)
from tests.mock_utils.mock_prompts import (
    MOCK_INSERT_PROMPT,
    MOCK_KEYWORD_EXTRACT_PROMPT,
    MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
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
        "table": {
            "keyword_extract_template": MOCK_KEYWORD_EXTRACT_PROMPT,
        },
    }
    query_configs = [
        QueryConfig(
            index_struct_type=IndexStructType.TREE,
            query_mode=QueryMode.DEFAULT,
            query_kwargs={
                "query_template": MOCK_QUERY_PROMPT,
                "text_qa_template": MOCK_TEXT_QA_PROMPT,
                "refine_template": MOCK_REFINE_PROMPT,
            },
        ),
        QueryConfig(
            index_struct_type=IndexStructType.LIST,
            query_mode=QueryMode.DEFAULT,
            query_kwargs={
                "text_qa_template": MOCK_TEXT_QA_PROMPT,
                "refine_template": MOCK_REFINE_PROMPT,
            },
        ),
        QueryConfig(
            index_struct_type=IndexStructType.KEYWORD_TABLE,
            query_mode=QueryMode.DEFAULT,
            query_kwargs={
                "query_keyword_extract_template": MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
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
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
def test_recursive_query_list_tree(
    _mock_init: Any,
    _mock_total_tokens_used: Any,
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


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
def test_recursive_query_tree_list(
    _mock_init: Any,
    _mock_total_tokens_used: Any,
    _mock_predict: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test query."""
    index_kwargs, query_configs = struct_kwargs
    list_kwargs = index_kwargs["list"]
    tree_kwargs = index_kwargs["tree"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    tree1 = GPTTreeIndex(documents[2:6], **tree_kwargs)
    tree2 = GPTTreeIndex(documents[:2] + documents[6:], **tree_kwargs)
    tree1.set_text("tree_summary1")
    tree2.set_text("tree_summary2")

    # there are two root nodes in this tree: one containing [list1, list2]
    # and the other containing [list3, list4]
    list_index = GPTListIndex([tree1, tree2], **list_kwargs)
    query_str = "What is?"
    # query should first pick the left root node, then pick list1
    # within list1, it should go through the first document and second document
    response = list_index.query(
        query_str, mode="recursive", query_configs=query_configs
    )
    assert response == ("What is?:This is a test.")

    # Also test a non-recursive query. This should not go down into the list
    list_query_kwargs = query_configs[1].query_kwargs
    response = list_index.query(query_str, mode="default", **list_query_kwargs)
    assert response == ("What is?:tree_summary1")


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
def test_recursive_query_table_list(
    _mock_init: Any,
    _mock_total_tokens_used: Any,
    _mock_predict: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test query."""
    index_kwargs, query_configs = struct_kwargs
    list_kwargs = index_kwargs["list"]
    table_kwargs = index_kwargs["table"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    table1 = GPTSimpleKeywordTableIndex(documents[4:6], **table_kwargs)
    table2 = GPTSimpleKeywordTableIndex(documents[2:3], **table_kwargs)
    table1.set_text("table_summary1")
    table2.set_text("table_summary2")

    list_index = GPTListIndex([table1, table2], **list_kwargs)
    query_str = "World?"
    response = list_index.query(
        query_str, mode="recursive", query_configs=query_configs
    )
    assert response == ("World?:Hello world.")

    query_str = "Test?"
    response = list_index.query(
        query_str, mode="recursive", query_configs=query_configs
    )
    assert response == ("Test?:This is a test.")


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
def test_recursive_query_list_table(
    _mock_init: Any,
    _mock_total_tokens_used: Any,
    _mock_predict: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test query."""
    index_kwargs, query_configs = struct_kwargs
    list_kwargs = index_kwargs["list"]
    table_kwargs = index_kwargs["table"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    # try building a list for every two, then a tree
    list1 = GPTListIndex(documents[0:2], **list_kwargs)
    list1.set_text("foo bar")
    list2 = GPTListIndex(documents[2:4], **list_kwargs)
    list2.set_text("apple orange")
    list3 = GPTListIndex(documents[4:6], **list_kwargs)
    list3.set_text("toronto london")
    list4 = GPTListIndex(documents[6:8], **list_kwargs)
    list4.set_text("cat dog")

    table = GPTSimpleKeywordTableIndex([list1, list2, list3, list4], **table_kwargs)
    query_str = "Foo?"
    response = table.query(query_str, mode="recursive", query_configs=query_configs)
    assert response == ("Foo?:This is a test v2.")
    query_str = "Orange?"
    response = table.query(query_str, mode="recursive", query_configs=query_configs)
    assert response == ("Orange?:This is a test.")
    query_str = "Cat?"
    response = table.query(query_str, mode="recursive", query_configs=query_configs)
    assert response == ("Cat?:This is another test.")


@patch.object(LLMChain, "predict", side_effect=mock_llmchain_predict)
@patch("gpt_index.langchain_helpers.chain_wrapper.OpenAI")
@patch.object(LLMChain, "__init__", return_value=None)
def test_recursive_query_list_tree_token_count(
    _mock_init: Any,
    _mock_llmchain: Any,
    _mock_predict: Any,
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
    # import pdb; pdb.set_trace()
    tree = GPTTreeIndex(
        [
            list1,
            list2,
            list3,
            list4,
        ],
        **tree_kwargs
    )
    # first pass prompt is "summary1\nsummary2\n" (6 tokens),
    # response is the mock response (10 tokens)
    # total is 16 tokens, multiply by 2 to get the total
    assert tree._llm_predictor.total_tokens_used == 32

    query_str = "What is?"
    # query should first pick the left root node, then pick list1
    # within list1, it should go through the first document and second document
    start_token_ct = tree._llm_predictor.total_tokens_used
    tree.query(query_str, mode="recursive", query_configs=query_configs)
    # prompt is which is 35 tokens, plus 10 for the mock response
    assert tree._llm_predictor.total_tokens_used - start_token_ct == 45
