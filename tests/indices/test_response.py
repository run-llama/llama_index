"""Test response utils."""

from typing import Any, List

import pytest

from gpt_index.constants import MAX_CHUNK_OVERLAP, MAX_CHUNK_SIZE, NUM_OUTPUTS
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.response.builder import ResponseBuilder, TextChunk
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_prompts import MOCK_REFINE_PROMPT, MOCK_TEXT_QA_PROMPT


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


@patch_common
def test_give_response(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test give response."""
    prompt_helper = PromptHelper(MAX_CHUNK_SIZE, NUM_OUTPUTS, MAX_CHUNK_OVERLAP)
    llm_predictor = LLMPredictor()
    query_str = "What is?"

    # test single line
    builder = ResponseBuilder(
        prompt_helper,
        llm_predictor,
        MOCK_TEXT_QA_PROMPT,
        MOCK_REFINE_PROMPT,
        texts=[TextChunk("This is a single line.")],
    )
    response = builder.get_response(query_str)
    assert response == "What is?:This is a single line."

    # test multiple lines
    builder = ResponseBuilder(
        prompt_helper,
        llm_predictor,
        MOCK_TEXT_QA_PROMPT,
        MOCK_REFINE_PROMPT,
        texts=[TextChunk(documents[0].get_text())],
    )
    response = builder.get_response(query_str)
    assert response == "What is?:Hello world."
