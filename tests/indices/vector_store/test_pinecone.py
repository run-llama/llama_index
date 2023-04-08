"""Test pinecone indexes."""

import sys
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.vector_store.vector_indices import GPTPineconeIndex

from gpt_index.readers.schema.base import Document
from tests.indices.vector_store.utils import MockPineconeIndex
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_prompts import MOCK_REFINE_PROMPT, MOCK_TEXT_QA_PROMPT

from tests.mock_utils.mock_utils import mock_tokenizer


@pytest.fixture
def struct_kwargs() -> Tuple[Dict, Dict]:
    """Index kwargs."""
    index_kwargs: Dict[str, Any] = {}
    query_kwargs: Dict[str, Any] = {
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
    elif text == "This is bar test.":
        return [0, 0, 1, 0, 0]
    elif text == "Hello world backup.":
        # this is used when "Hello world." is deleted.
        return [1, 0, 0, 0, 0]
    else:
        raise ValueError("Invalid text for `mock_get_text_embedding`.")


def mock_get_text_embeddings(texts: List[str]) -> List[List[float]]:
    """Mock get text embeddings."""
    return [mock_get_text_embedding(text) for text in texts]


def mock_get_query_embedding(query: str) -> List[float]:
    """Mock get query embedding."""
    return [0, 0, 1, 0, 0]


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "get_query_embedding", side_effect=mock_get_query_embedding
)
def test_build_pinecone(
    _mock_query_embed: Any,
    _mock_embeds: Any,
    _mock_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test build GPTPineconeIndex."""
    # NOTE: mock pinecone import
    sys.modules["pinecone"] = MagicMock()
    # NOTE: mock pinecone index
    pinecone_index = MockPineconeIndex()

    index_kwargs, query_kwargs = struct_kwargs

    index = GPTPineconeIndex.from_documents(
        documents=documents,
        pinecone_index=pinecone_index,
        tokenizer=mock_tokenizer,
        **index_kwargs
    )

    response = index.query("What is?", **query_kwargs)
    assert str(response) == ("What is?:This is another test.")
