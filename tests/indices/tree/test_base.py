"""Test tree index."""

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from gpt_index.indices.data_structs import IndexGraph, Node
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMChain
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_predict import mock_llmchain_predict
from tests.mock_utils.mock_prompts import (
    MOCK_INSERT_PROMPT,
    MOCK_QUERY_PROMPT,
    MOCK_REFINE_PROMPT,
    MOCK_SUMMARY_PROMPT,
    MOCK_TEXT_QA_PROMPT,
)


@pytest.fixture
def struct_kwargs() -> Tuple[Dict, Dict]:
    """Index kwargs."""
    index_kwargs = {
        "summary_template": MOCK_SUMMARY_PROMPT,
        "insert_prompt": MOCK_INSERT_PROMPT,
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


def _get_left_or_right_node(
    index_graph: IndexGraph, node: Optional[Node], left: bool = True
) -> Node:
    """Get 'left' or 'right' node."""
    if node is None:
        indices = set(index_graph.root_nodes.keys())
    else:
        indices = node.child_indices

    index = min(indices) if left else max(indices)

    if index not in index_graph.all_nodes:
        raise ValueError(f"Node {index} not in index_graph.all_nodes")
    return index_graph.all_nodes[index]


@patch_common
def test_build_tree(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
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


@patch_common
def test_build_tree_multiple(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test build tree."""
    new_docs = [
        Document("Hello world.\nThis is a test."),
        Document("This is another test.\nThis is a test v2."),
    ]
    index_kwargs, _ = struct_kwargs
    tree = GPTTreeIndex(new_docs, **index_kwargs)
    assert len(tree.index_struct.all_nodes) == 6
    # check contents of nodes
    assert tree.index_struct.all_nodes[0].text == "Hello world."
    assert tree.index_struct.all_nodes[1].text == "This is a test."
    assert tree.index_struct.all_nodes[2].text == "This is another test."
    assert tree.index_struct.all_nodes[3].text == "This is a test v2."


@patch_common
def test_query(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test query."""
    index_kwargs, query_kwargs = struct_kwargs
    tree = GPTTreeIndex(documents, **index_kwargs)

    # test default query
    query_str = "What is?"
    response = tree.query(query_str, mode="default", **query_kwargs)
    assert response == ("What is?:Hello world.")


@patch_common
def test_summarize_query(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test summarize query."""
    # create tree index without building tree
    index_kwargs, orig_query_kwargs = struct_kwargs
    index_kwargs = index_kwargs.copy()
    index_kwargs.update({"build_tree": False})
    tree = GPTTreeIndex(documents, **index_kwargs)

    # test summarize query
    query_str = "What is?"
    query_kwargs: Dict[str, Any] = {
        "text_qa_template": MOCK_TEXT_QA_PROMPT,
        "num_children": 2,
    }
    # TODO: fix unit test later
    response = tree.query(query_str, mode="summarize", **query_kwargs)
    assert response == ("What is?:Hello world.")

    # test that default query fails
    with pytest.raises(ValueError):
        tree.query(query_str, mode="default", **orig_query_kwargs)


@patch_common
def test_insert(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test insert."""
    index_kwargs, _ = struct_kwargs
    tree = GPTTreeIndex(documents, **index_kwargs)

    # test insert
    new_doc = Document("This is a new doc.", doc_id="new_doc")
    tree.insert(new_doc)
    # Before:
    # Left root node: "Hello world.\nThis is a test."
    # "Hello world.", "This is a test" are two children of the left root node
    # After:
    # "Hello world.\nThis is a test\n.\nThis is a new doc." is the left root node
    # "Hello world", "This is a test\n.This is a new doc." are the children
    # of the left root node.
    # "This is a test", "This is a new doc." are the children of
    # "This is a test\n.This is a new doc."
    left_root = _get_left_or_right_node(tree.index_struct, None)
    assert left_root.text == "Hello world.\nThis is a test.\nThis is a new doc."
    left_root2 = _get_left_or_right_node(tree.index_struct, left_root)
    right_root2 = _get_left_or_right_node(tree.index_struct, left_root, left=False)
    assert left_root2.text == "Hello world."
    assert right_root2.text == "This is a test.\nThis is a new doc."
    left_root3 = _get_left_or_right_node(tree.index_struct, right_root2)
    right_root3 = _get_left_or_right_node(tree.index_struct, right_root2, left=False)
    assert left_root3.text == "This is a test."
    assert right_root3.text == "This is a new doc."
    assert right_root3.ref_doc_id == "new_doc"

    # test insert from empty (no_id)
    tree = GPTTreeIndex([], **index_kwargs)
    new_doc = Document("This is a new doc.")
    tree.insert(new_doc)
    assert len(tree.index_struct.all_nodes) == 1
    assert tree.index_struct.all_nodes[0].text == "This is a new doc."

    # test insert from empty (with_id)
    tree = GPTTreeIndex([], **index_kwargs)
    new_doc = Document("This is a new doc.", doc_id="new_doc_test")
    tree.insert(new_doc)
    assert len(tree.index_struct.all_nodes) == 1
    assert tree.index_struct.all_nodes[0].text == "This is a new doc."
    assert tree.index_struct.all_nodes[0].ref_doc_id == "new_doc_test"


@patch.object(LLMChain, "predict", side_effect=mock_llmchain_predict)
@patch("gpt_index.langchain_helpers.chain_wrapper.OpenAI")
@patch.object(LLMChain, "__init__", return_value=None)
def test_build_and_count_tokens(
    _mock_init: Any,
    _mock_llmchain: Any,
    _mock_predict: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test build and count tokens."""
    index_kwargs, _ = struct_kwargs
    # mock_prompts.MOCK_SUMMARY_PROMPT_TMPL adds a "\n" to the document text
    # and the document is 23 tokens
    document_token_count = 24
    llmchain_mock_resp_token_count = 10
    tree = GPTTreeIndex(documents, **index_kwargs)
    assert (
        tree._llm_predictor.total_tokens_used
        == document_token_count + llmchain_mock_resp_token_count
    )
