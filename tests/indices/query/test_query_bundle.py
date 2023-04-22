"""Test query bundle."""

from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Correct.\n"
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(doc_text)]


def _get_query_embedding(
    query: str,
) -> List[float]:
    """Get query embedding."""
    text_embed_map: Dict[str, List[float]] = {
        "It is what it is.": [1.0, 0.0, 0.0, 0.0, 0.0],
        "The meaning of life": [0.0, 1.0, 0.0, 0.0, 0.0],
    }

    return text_embed_map[query]


def _get_text_embedding(
    text: str,
) -> List[float]:
    """Get node text embedding."""
    text_embed_map: Dict[str, List[float]] = {
        "Correct.": [0.5, 0.5, 0.0, 0.0, 0.0],
        "Hello world.": [1.0, 0.0, 0.0, 0.0, 0.0],
        "This is a test.": [0.0, 1.0, 0.0, 0.0, 0.0],
        "This is another test.": [0.0, 0.0, 1.0, 0.0, 0.0],
        "This is a test v2.": [0.0, 0.0, 0.0, 1.0, 0.0],
    }

    return text_embed_map[text]


def _get_text_embeddings(
    texts: List[str],
) -> List[List[float]]:
    """Get node text embedding."""
    return [_get_text_embedding(text) for text in texts]


@patch_common
@patch.object(
    OpenAIEmbedding,
    "_get_query_embedding",
    side_effect=_get_query_embedding,
)
@patch.object(
    OpenAIEmbedding,
    "_get_text_embedding",
    side_effect=_get_text_embedding,
)
@patch.object(OpenAIEmbedding, "_get_text_embeddings", side_effect=_get_text_embeddings)
def test_embedding_query(
    _mock_get_text_embedding: Any,
    _mock_get_text_embeddings: Any,
    _mock_get_query_embedding: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test embedding query."""
    index = GPTListIndex.from_documents(documents)

    # test embedding query
    query_bundle = QueryBundle(
        query_str="What is?",
        custom_embedding_strs=[
            "It is what it is.",
            "The meaning of life",
        ],
    )
    retriever = index.as_retriever(mode="embedding", similarity_top_k=1)
    nodes = retriever.retrieve(query_bundle)
    assert len(nodes) == 1
    assert nodes[0].node.text == "Correct."
