"""Test embedding functionalities."""

from collections import defaultdict
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from llama_index.core.indices.tree.base import TreeIndex
from llama_index.core.indices.tree.select_leaf_embedding_retriever import (
    TreeSelectLeafEmbeddingRetriever,
)
from llama_index.core.schema import BaseNode, Document, QueryBundle
from tests.mock_utils.mock_prompts import (
    MOCK_INSERT_PROMPT,
    MOCK_SUMMARY_PROMPT,
)


@pytest.fixture()
def index_kwargs() -> dict:
    """Index kwargs."""
    return {
        "summary_template": MOCK_SUMMARY_PROMPT,
        "insert_prompt": MOCK_INSERT_PROMPT,
        "num_children": 2,
    }


@pytest.fixture()
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\nThis is a test.\nThis is another test.\nThis is a test v2."
    )
    return [Document(text=doc_text)]


def _get_node_text_embedding_similarities(
    query_embedding: List[float], nodes: List[BaseNode]
) -> List[float]:
    """Get node text embedding similarity."""
    text_similarity_map = defaultdict(lambda: 0.0)
    text_similarity_map["Hello world."] = 0.9
    text_similarity_map["This is a test."] = 0.8
    text_similarity_map["This is another test."] = 0.7
    text_similarity_map["This is a test v2."] = 0.6

    similarities = []
    for node in nodes:
        similarities.append(text_similarity_map[node.get_content()])

    return similarities


@patch.object(
    TreeSelectLeafEmbeddingRetriever,
    "_get_query_text_embedding_similarities",
    side_effect=_get_node_text_embedding_similarities,
)
def test_embedding_query(
    _patch_similarity: Any,
    index_kwargs: Dict,
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
) -> None:
    """Test embedding query."""
    tree = TreeIndex.from_documents(documents, **index_kwargs)

    # test embedding query
    query_str = "What is?"
    retriever = tree.as_retriever(retriever_mode="select_leaf_embedding")
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert nodes[0].node.get_content() == "Hello world."
