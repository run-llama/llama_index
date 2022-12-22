"""Test Faiss index."""

import sys
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.vector_store.faiss import GPTFaissIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.readers.schema.base import Document
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
        "similarity_top_k": 1,
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


class MockFaissIndex:
    """Mock Faiss index."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize params."""
        self._index: Dict[int, np.ndarray] = {}

    @property
    def ntotal(self) -> int:
        """Get ntotal."""
        return len(self._index)

    def add(self, vecs: np.ndarray) -> None:
        """Add vectors to index."""
        for vec in vecs:
            new_id = len(self._index)
            self._index[new_id] = vec

    def search(self, vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search index."""
        # assume query vec is of the form 1 x k
        # index_mat is n x k
        index_mat = np.array(list(self._index.values()))
        # compute distances
        distances = np.linalg.norm(index_mat - vec, axis=1)

        indices = np.argsort(distances)[:k]
        sorted_distances = distances[indices][:k]

        # return distances and indices
        return sorted_distances[np.newaxis, :], indices[np.newaxis, :]


def mock_get_text_embedding(text: str) -> List[float]:
    """Mock get text embedding."""
    # assume dimensions are 5
    if text == "Hello world.":
        return [1, 0, 0, 0, 0]
    elif text == "This is a test.":
        return [0, 1, 0, 0, 0]
    elif text == "This is another test.":
        return [0, 0, 1, 0, 0]
    elif text == "This is a test v2.":
        return [0, 0, 0, 1, 0]
    elif text == "This is a test v3.":
        return [0, 0, 0, 0, 1]
    else:
        raise ValueError("Invalid text for `mock_get_text_embedding`.")


def mock_get_query_embedding(query: str) -> List[float]:
    """Mock get query embedding."""
    return [0, 0, 1, 0, 0]


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
@patch.object(
    OpenAIEmbedding, "get_text_embedding", side_effect=mock_get_text_embedding
)
def test_build_faiss(
    _mock_embed: Any,
    _mock_init: Any,
    _mock_total_tokens_used: Any,
    _mock_splitter: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test build GPTFaissIndex."""
    # NOTE: mock faiss import
    sys.modules["faiss"] = MagicMock()
    # NOTE: mock faiss index
    faiss_index = MockFaissIndex()

    index_kwargs, query_kwargs = struct_kwargs

    index = GPTFaissIndex(documents=documents, faiss_index=faiss_index, **index_kwargs)
    assert len(index.index_struct.nodes_dict) == 4
    # check contents of nodes
    assert index.index_struct.get_node(0).text == "Hello world."
    assert index.index_struct.get_node(1).text == "This is a test."
    assert index.index_struct.get_node(2).text == "This is another test."
    assert index.index_struct.get_node(3).text == "This is a test v2."


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
@patch.object(
    OpenAIEmbedding, "get_text_embedding", side_effect=mock_get_text_embedding
)
def test_faiss_insert(
    _mock_embed: Any,
    _mock_init: Any,
    _mock_total_tokens_used: Any,
    _mock_splitter: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test build GPTFaissIndex."""
    # NOTE: mock faiss import
    sys.modules["faiss"] = MagicMock()
    # NOTE: mock faiss index
    faiss_index = MockFaissIndex()

    index_kwargs, query_kwargs = struct_kwargs

    index = GPTFaissIndex(documents=documents, faiss_index=faiss_index, **index_kwargs)
    # insert into index
    index.insert(Document(text="This is a test v3."))

    # check contenst of nodes
    assert index.index_struct.get_node(3).text == "This is a test v2."
    assert index.index_struct.get_node(4).text == "This is a test v3."


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(
    OpenAIEmbedding, "get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "get_query_embedding", side_effect=mock_get_query_embedding
)
def test_embedding_query(
    _mock_query_embed: Any,
    _mock_text_embed: Any,
    _mock_predict: Any,
    _mock_init: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test embedding query."""
    # NOTE: mock faiss import
    sys.modules["faiss"] = MagicMock()
    # NOTE: mock faiss index
    faiss_index = MockFaissIndex()

    index_kwargs, query_kwargs = struct_kwargs
    index = GPTFaissIndex(documents, faiss_index=faiss_index, **index_kwargs)

    # test embedding query
    query_str = "What is?"
    response = index.query(query_str, **query_kwargs)
    assert response == ("What is?:This is another test.")
