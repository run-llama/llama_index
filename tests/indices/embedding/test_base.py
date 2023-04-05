"""Test embedding functionalities."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

from gpt_index.data_structs.node_v2 import Node
from gpt_index.embeddings.base import mean_agg
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.tree.embedding_query import GPTTreeIndexEmbeddingQuery
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.langchain_helpers.chain_wrapper import (
    LLMChain,
    LLMMetadata,
    LLMPredictor,
)
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
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
from tests.mock_utils.mock_text_splitter import (
    mock_token_splitter_newline_with_overlaps,
)


def test_embedding_similarity() -> None:
    """Test embedding similarity."""
    embed_model = OpenAIEmbedding()
    text_embedding = [3.0, 4.0, 0.0]
    query_embedding = [0.0, 1.0, 0.0]
    cosine = embed_model.similarity(query_embedding, text_embedding)
    assert cosine == 0.8


def test_mean_agg() -> None:
    """Test mean aggregation for embeddings."""
    embedding_0 = [3.0, 4.0, 0.0]
    embedding_1 = [0.0, 1.0, 0.0]
    output = mean_agg([embedding_0, embedding_1])
    assert output == [1.5, 2.5, 0.0]


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
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Dict,
    documents: List[Document],
) -> None:
    """Test embedding query."""
    index_kwargs, query_kwargs = struct_kwargs
    tree = GPTTreeIndex.from_documents(documents, **index_kwargs)

    # test embedding query
    query_str = "What is?"
    response = tree.query(query_str, mode="embedding", **query_kwargs)
    assert str(response) == ("What is?:Hello world.")


def _mock_tokenizer(text: str) -> int:
    """Mock tokenizer that splits by spaces."""
    return len(text.split(" "))


@patch.object(LLMChain, "predict", side_effect=mock_llmchain_predict)
@patch("gpt_index.llm_predictor.base.OpenAI")
@patch.object(LLMPredictor, "get_llm_metadata", return_value=LLMMetadata())
@patch.object(LLMChain, "__init__", return_value=None)
@patch.object(
    GPTTreeIndexEmbeddingQuery,
    "_get_query_text_embedding_similarities",
    side_effect=_get_node_text_embedding_similarities,
)
@patch.object(
    TokenTextSplitter,
    "split_text_with_overlaps",
    side_effect=mock_token_splitter_newline_with_overlaps,
)
@patch.object(LLMPredictor, "_count_tokens", side_effect=_mock_tokenizer)
def test_query_and_count_tokens(
    _mock_count_tokens: Any,
    _mock_split_text: Any,
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
    # First block is "Hello world.\nThis is a test.\n"
    # Second block is "This is another test.\nThis is a test v2."
    # first block is 5 tokens because
    # last word of first line and first word of second line are joined
    # second block is 8 tokens for similar reasons.
    first_block_count = 5
    second_block_count = 8
    llmchain_mock_resp_token_count = 4
    # build the tree
    # TMP
    tree = GPTTreeIndex.from_documents(documents, **index_kwargs)
    assert tree.service_context.llm_predictor.total_tokens_used == (
        first_block_count + llmchain_mock_resp_token_count
    ) + (second_block_count + llmchain_mock_resp_token_count)

    # test embedding query
    start_token_ct = tree._service_context.llm_predictor.total_tokens_used
    query_str = "What is?"
    # context is "hello world." which is 2 tokens
    context_tokens = 2
    # query is "what is?" which is 2 tokens
    query_tokens = 2
    # subtract one because the last token of the context is joined with first
    input_tokens = context_tokens + query_tokens - 1

    tree.query(query_str, mode="embedding", **query_kwargs)
    assert (
        tree.service_context.llm_predictor.total_tokens_used - start_token_ct
        == input_tokens + llmchain_mock_resp_token_count
    )
