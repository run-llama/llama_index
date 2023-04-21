from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.llm_predictor.base import LLMPredictor
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_predict import mock_llmpredictor_predict
from tests.mock_utils.mock_prompts import MOCK_TEXT_QA_PROMPT


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
    retriever = tree.as_retriever(mode="default")
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) == 1


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
    # TODO: fix unit test later
    retriever = tree.as_retriever(mode="summarize")
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) == 4
