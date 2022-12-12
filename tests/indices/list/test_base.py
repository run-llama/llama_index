"""Test list index."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

from gpt_index.indices.data_structs import Node
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.query.list.embedding_query import GPTListIndexEmbeddingQuery
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.schema import Document
from tests.mock_utils.mock_predict import mock_llmpredictor_predict
from tests.mock_utils.mock_prompts import MOCK_REFINE_PROMPT, MOCK_TEXT_QA_PROMPT
from tests.mock_utils.mock_text_splitter import mock_token_splitter_newline


@pytest.fixture
def struct_kwargs() -> Tuple[Dict, Dict]:
    """Index kwargs."""
    index_kwargs = {
        "text_qa_template": MOCK_TEXT_QA_PROMPT,
    }
    query_kwargs = {
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
@patch.object(LLMPredictor, "__init__", return_value=None)
def test_build_list(
    _mock_init: Any, _mock_splitter: Any, documents: List[Document]
) -> None:
    """Test build list."""
    list_index = GPTListIndex(documents=documents)
    assert len(list_index.index_struct.nodes) == 4
    # check contents of nodes
    assert list_index.index_struct.nodes[0].text == "Hello world."
    assert list_index.index_struct.nodes[1].text == "This is a test."
    assert list_index.index_struct.nodes[2].text == "This is another test."
    assert list_index.index_struct.nodes[3].text == "This is a test v2."


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "__init__", return_value=None)
def test_build_list_multiple(_mock_init: Any, _mock_splitter: Any) -> None:
    """Test build list multiple."""
    documents = [
        Document("Hello world.\nThis is a test."),
        Document("This is another test.\nThis is a test v2."),
    ]
    list_index = GPTListIndex(documents=documents)
    assert len(list_index.index_struct.nodes) == 4
    # check contents of nodes
    assert list_index.index_struct.nodes[0].text == "Hello world."
    assert list_index.index_struct.nodes[1].text == "This is a test."
    assert list_index.index_struct.nodes[2].text == "This is another test."
    assert list_index.index_struct.nodes[3].text == "This is a test v2."


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
def test_list_insert(
    _mock_init: Any,
    _mock_total_tokens_used: Any,
    _mock_splitter: Any,
    documents: List[Document],
) -> None:
    """Test insert to list."""
    list_index = GPTListIndex([])
    assert len(list_index.index_struct.nodes) == 0
    list_index.insert(documents[0])
    # check contents of nodes
    assert list_index.index_struct.nodes[0].text == "Hello world."
    assert list_index.index_struct.nodes[1].text == "This is a test."
    assert list_index.index_struct.nodes[2].text == "This is another test."
    assert list_index.index_struct.nodes[3].text == "This is a test v2."

    # test insert with ID
    document = documents[0]
    document.doc_id = "test_id"
    list_index = GPTListIndex([])
    list_index.insert(document)
    # check contents of nodes
    for node in list_index.index_struct.nodes:
        assert node.ref_doc_id == "test_id"


def _get_node_text_embedding_similarities(
    query_embedding: List[float], nodes: List[Node]
) -> List[float]:
    """Get node text embedding similarity."""
    text_similarity_map = defaultdict(lambda: 0.0)
    text_similarity_map["Hello world."] = 0.9
    text_similarity_map["This is a test."] = 0.8
    text_similarity_map["This is another test."] = 0.7
    text_similarity_map["This is a test v2."] = 0.6

    similarities = []
    for node in nodes:
        similarities.append(text_similarity_map[node.get_text()])

    return similarities


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
def test_query(
    _mock_predict: Any,
    _mock_init: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test embedding query."""
    index_kwargs, query_kwargs = struct_kwargs
    index = GPTListIndex(documents, **index_kwargs)

    # test embedding query
    query_str = "What is?"
    response = index.query(query_str, mode="default", **query_kwargs)
    assert response == ("What is?:Hello world.")


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(
    GPTListIndexEmbeddingQuery,
    "_get_query_text_embedding_similarities",
    side_effect=_get_node_text_embedding_similarities,
)
def test_embedding_query(
    _mock_similarity: Any,
    _mock_predict: Any,
    _mock_init: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test embedding query."""
    index_kwargs, query_kwargs = struct_kwargs
    index = GPTListIndex(documents, **index_kwargs)

    # test embedding query
    query_str = "What is?"
    response = index.query(
        query_str, mode="embedding", similarity_top_k=1, **query_kwargs
    )
    assert response == ("What is?:Hello world.")
