"""Test embedding functionalities."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.data_structs import Node
from gpt_index.indices.query.tree.embedding_query import GPTTreeIndexEmbeddingQuery
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.langchain_helpers.chain_wrapper import (
    LLMChain,
    LLMMetadata,
    LLMPredictor,
)
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


def test_embedding_similarity() -> None:
    """Test embedding similarity."""
    embed_model = OpenAIEmbedding()
    text_embedding = [3.0, 4.0, 0.0]
    query_embedding = [0.0, 1.0, 0.0]
    cosine = embed_model.similarity(query_embedding, text_embedding)
    assert cosine == 0.8


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


@patch_common
@patch.object(
    GPTTreeIndexEmbeddingQuery,
    "_get_query_text_embedding_similarities",
    side_effect=_get_node_text_embedding_similarities,
)
def test_embedding_query(
    _mock_similarity: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    struct_kwargs: Dict,
    documents: List[Document],
) -> None:
    """Test embedding query."""
    index_kwargs, query_kwargs = struct_kwargs
    tree = GPTTreeIndex(documents, **index_kwargs)

    # test embedding query
    query_str = "What is?"
    response = tree.query(query_str, mode="embedding", **query_kwargs)
    assert response == ("What is?:Hello world.")


@patch.object(LLMChain, "predict", side_effect=mock_llmchain_predict)
@patch("gpt_index.langchain_helpers.chain_wrapper.OpenAI")
@patch.object(LLMPredictor, "get_llm_metadata", return_value=LLMMetadata())
@patch.object(LLMChain, "__init__", return_value=None)
@patch.object(
    GPTTreeIndexEmbeddingQuery,
    "_get_query_text_embedding_similarities",
    side_effect=_get_node_text_embedding_similarities,
)
def test_query_and_count_tokens(
    _mock_similarity: Any,
    _mock_llmchain: Any,
    _mock_llm_metadata: Any,
    _mock_init: Any,
    _mock_predict: Any,
    struct_kwargs: Dict,
    documents: List[Document],
) -> None:
    """Test query and count tokens."""
    index_kwargs, query_kwargs = struct_kwargs
    # mock_prompts.MOCK_SUMMARY_PROMPT_TMPL adds a "\n" to the document text
    document_token_count = 24
    llmchain_mock_resp_token_count = 10
    # build the tree
    tree = GPTTreeIndex(documents, **index_kwargs)
    assert (
        tree._llm_predictor.total_tokens_used
        == document_token_count + llmchain_mock_resp_token_count
    )

    # test embedding query
    start_token_ct = tree._llm_predictor.total_tokens_used
    query_str = "What is?"
    # From MOCK_TEXT_QA_PROMPT, the prompt is 28 total
    query_prompt_token_count = 28
    tree.query(query_str, mode="embedding", **query_kwargs)
    assert (
        tree._llm_predictor.total_tokens_used - start_token_ct
        == query_prompt_token_count + llmchain_mock_resp_token_count
    )
