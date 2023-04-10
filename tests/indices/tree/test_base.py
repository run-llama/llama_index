"""Test tree index."""

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from gpt_index.data_structs.data_structs_v2 import IndexGraph
from gpt_index.data_structs.node_v2 import Node
from gpt_index.docstore import DocumentStore
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.langchain_helpers.chain_wrapper import (
    LLMChain,
    LLMMetadata,
    LLMPredictor,
)
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_predict import (
    mock_llmchain_predict,
    mock_llmpredictor_predict,
)
from tests.mock_utils.mock_prompts import (
    MOCK_INSERT_PROMPT,
    MOCK_QUERY_PROMPT,
    MOCK_REFINE_PROMPT,
    MOCK_SUMMARY_PROMPT,
    MOCK_TEXT_QA_PROMPT,
)
from tests.mock_utils.mock_text_splitter import (
    mock_token_splitter_newline_with_overlaps,
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
    docstore: DocumentStore,
    index_graph: IndexGraph,
    node: Optional[Node],
    left: bool = True,
) -> Node:
    """Get 'left' or 'right' node."""
    children_dict = index_graph.get_children(node)
    indices = list(children_dict.keys())
    index = min(indices) if left else max(indices)
    node_id = children_dict[index]
    return docstore.get_node(node_id)


@patch_common
def test_build_tree(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test build tree."""
    index_kwargs, _ = struct_kwargs
    tree = GPTTreeIndex.from_documents(documents, **index_kwargs)
    assert len(tree.index_struct.all_nodes) == 6
    # check contents of nodes

    print(tree.docstore.docs)
    print(len(tree.docstore.docs))
    print(tree.index_struct.all_nodes)
    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert nodes[0].text == "Hello world."
    assert nodes[1].text == "This is a test."
    assert nodes[2].text == "This is another test."
    assert nodes[3].text == "This is a test v2."
    assert nodes[4].text == ("Hello world.\nThis is a test.")
    assert nodes[5].text == ("This is another test.\nThis is a test v2.")


@patch_common
def test_build_tree_with_embed(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test build tree."""
    index_kwargs, _ = struct_kwargs
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    document = Document(doc_text, embedding=[0.1, 0.2, 0.3])
    tree = GPTTreeIndex.from_documents([document], **index_kwargs)
    assert len(tree.index_struct.all_nodes) == 6
    # check contents of nodes
    all_nodes = tree.docstore.get_node_dict(tree.index_struct.all_nodes)
    assert all_nodes[0].text == "Hello world."
    assert all_nodes[1].text == "This is a test."
    assert all_nodes[2].text == "This is another test."
    assert all_nodes[3].text == "This is a test v2."
    # make sure all leaf nodes have embeddings
    for i in range(4):
        assert all_nodes[i].embedding == [0.1, 0.2, 0.3]
    assert all_nodes[4].text == ("Hello world.\nThis is a test.")
    assert all_nodes[5].text == ("This is another test.\nThis is a test v2.")


OUTPUTS = [
    ("Hello world.\nThis is a test.", ""),
    ("This is another test.\nThis is a test v2.", ""),
]


@patch_common
@patch.object(LLMPredictor, "apredict", side_effect=mock_llmpredictor_predict)
@patch("gpt_index.indices.common_tree.base.run_async_tasks", side_effect=[OUTPUTS])
def test_build_tree_async(
    _mock_run_async_tasks: Any,
    _mock_apredict: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test build tree with use_async."""
    index_kwargs, _ = struct_kwargs
    tree = GPTTreeIndex.from_documents(documents, use_async=True, **index_kwargs)
    assert len(tree.index_struct.all_nodes) == 6
    # check contents of nodes
    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert nodes[0].text == "Hello world."
    assert nodes[1].text == "This is a test."
    assert nodes[2].text == "This is another test."
    assert nodes[3].text == "This is a test v2."
    assert nodes[4].text == ("Hello world.\nThis is a test.")
    assert nodes[5].text == ("This is another test.\nThis is a test v2.")


@patch_common
def test_build_tree_multiple(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
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
    tree = GPTTreeIndex.from_documents(new_docs, **index_kwargs)
    assert len(tree.index_struct.all_nodes) == 6
    # check contents of nodes
    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert nodes[0].text == "Hello world."
    assert nodes[1].text == "This is a test."
    assert nodes[2].text == "This is another test."
    assert nodes[3].text == "This is a test v2."


@patch_common
def test_query(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test query."""
    index_kwargs, query_kwargs = struct_kwargs
    tree = GPTTreeIndex.from_documents(documents, **index_kwargs)

    # test default query
    query_str = "What is?"
    response = tree.query(query_str, mode="default", **query_kwargs)
    assert str(response) == ("What is?:Hello world.")


@patch_common
@patch.object(LLMPredictor, "apredict", side_effect=mock_llmpredictor_predict)
def test_summarize_query(
    _mock_apredict: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test summarize query."""
    # create tree index without building tree
    index_kwargs, orig_query_kwargs = struct_kwargs
    index_kwargs = index_kwargs.copy()
    index_kwargs.update({"build_tree": False})
    tree = GPTTreeIndex.from_documents(documents, **index_kwargs)

    # test summarize query
    query_str = "What is?"
    query_kwargs: Dict[str, Any] = {
        "text_qa_template": MOCK_TEXT_QA_PROMPT,
        "num_children": 2,
    }
    # TODO: fix unit test later
    response = tree.query(query_str, mode="summarize", **query_kwargs)
    print(str(response))
    assert str(response) == (
        "What is?:Hello world.:This is a test.:This is another test.:This is a test v2."
    )

    # test that default query fails
    with pytest.raises(ValueError):
        tree.query(query_str, mode="default", **orig_query_kwargs)


@patch_common
def test_insert(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test insert."""
    index_kwargs, _ = struct_kwargs
    tree = GPTTreeIndex.from_documents(documents, **index_kwargs)

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
    left_root = _get_left_or_right_node(tree.docstore, tree.index_struct, None)
    assert left_root.text == "Hello world.\nThis is a test.\nThis is a new doc."
    left_root2 = _get_left_or_right_node(tree.docstore, tree.index_struct, left_root)
    right_root2 = _get_left_or_right_node(
        tree.docstore, tree.index_struct, left_root, left=False
    )
    assert left_root2.text == "Hello world."
    assert right_root2.text == "This is a test.\nThis is a new doc."
    left_root3 = _get_left_or_right_node(tree.docstore, tree.index_struct, right_root2)
    right_root3 = _get_left_or_right_node(
        tree.docstore, tree.index_struct, right_root2, left=False
    )
    assert left_root3.text == "This is a test."
    assert right_root3.text == "This is a new doc."
    assert right_root3.ref_doc_id == "new_doc"

    # test insert from empty (no_id)
    tree = GPTTreeIndex.from_documents([], **index_kwargs)
    new_doc = Document("This is a new doc.")
    tree.insert(new_doc)
    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert len(nodes) == 1
    assert nodes[0].text == "This is a new doc."

    # test insert from empty (with_id)
    tree = GPTTreeIndex.from_documents([], **index_kwargs)
    new_doc = Document("This is a new doc.", doc_id="new_doc_test")
    tree.insert(new_doc)
    assert len(tree.index_struct.all_nodes) == 1
    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert nodes[0].text == "This is a new doc."
    assert nodes[0].ref_doc_id == "new_doc_test"


def _mock_tokenizer(text: str) -> int:
    """Mock tokenizer that splits by spaces."""
    return len(text.split(" "))


@patch.object(LLMChain, "predict", side_effect=mock_llmchain_predict)
@patch("gpt_index.llm_predictor.base.OpenAI")
@patch.object(LLMPredictor, "get_llm_metadata", return_value=LLMMetadata())
@patch.object(LLMChain, "__init__", return_value=None)
@patch.object(
    TokenTextSplitter,
    "split_text_with_overlaps",
    side_effect=mock_token_splitter_newline_with_overlaps,
)
@patch.object(LLMPredictor, "_count_tokens", side_effect=_mock_tokenizer)
def test_build_and_count_tokens(
    _mock_count_tokens: Any,
    _mock_split_text: Any,
    _mock_init: Any,
    _mock_llm_metadata: Any,
    _mock_llmchain: Any,
    _mock_predict: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test build and count tokens."""
    index_kwargs, _ = struct_kwargs
    # First block is "Hello world.\nThis is a test.\n"
    # Second block is "This is another test.\nThis is a test v2."
    # first block is 5 tokens because
    # last word of first line and first word of second line are joined
    # second block is 8 tokens for similar reasons.
    first_block_count = 5
    second_block_count = 8
    llmchain_mock_resp_token_count = 4
    tree = GPTTreeIndex.from_documents(documents, **index_kwargs)
    assert tree.service_context.llm_predictor.total_tokens_used == (
        (first_block_count + llmchain_mock_resp_token_count)
        + (second_block_count + llmchain_mock_resp_token_count)
    )
