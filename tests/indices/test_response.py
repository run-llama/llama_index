"""Test response utils."""

from typing import Any, List
from unittest.mock import patch

import pytest

from gpt_index.constants import MAX_CHUNK_OVERLAP, MAX_CHUNK_SIZE, NUM_OUTPUTS
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.response.builder import ResponseBuilder, ResponseMode, TextChunk
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_predict import mock_llmpredictor_predict
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


def mock_tokenizer(text: str) -> List[str]:
    """Mock tokenizer."""
    if text == "":
        return []
    tokens = text.split(" ")
    return tokens


@patch_common
def test_give_response(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
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
    assert str(response) == "What is?:This is a single line."

    # test multiple lines
    builder = ResponseBuilder(
        prompt_helper,
        llm_predictor,
        MOCK_TEXT_QA_PROMPT,
        MOCK_REFINE_PROMPT,
        texts=[TextChunk(documents[0].get_text())],
    )
    response = builder.get_response(query_str)
    assert str(response) == "What is?:Hello world."


@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(LLMPredictor, "__init__", return_value=None)
def test_compact_response(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    documents: List[Document],
) -> None:
    """Test give response."""
    # test response with ResponseMode.COMPACT
    # NOTE: here we want to guarante that prompts have 0 extra tokens
    mock_refine_prompt_tmpl = "{query_str}{existing_answer}{context_msg}"
    mock_refine_prompt = RefinePrompt(mock_refine_prompt_tmpl)

    mock_qa_prompt_tmpl = "{context_str}{query_str}"
    mock_qa_prompt = QuestionAnswerPrompt(mock_qa_prompt_tmpl)

    # max input size is 11, prompt is two tokens (the query) --> 9 tokens
    # --> padding is 1 --> 8 tokens
    prompt_helper = PromptHelper(
        11, 0, 0, tokenizer=mock_tokenizer, separator="\n\n", chunk_size_limit=4
    )
    cur_chunk_size = prompt_helper.get_chunk_size_given_prompt("", 1, padding=1)
    # outside of compact, assert that chunk size is 4
    assert cur_chunk_size == 4

    # within compact, make sure that chunk size is 8
    llm_predictor = LLMPredictor()
    query_str = "What is?"
    texts = [
        TextChunk("This\n\nis\n\na\n\nbar"),
        TextChunk("This\n\nis\n\na\n\ntest"),
        TextChunk("This\n\nis\n\nanother\n\ntest"),
        TextChunk("This\n\nis\n\na\n\nfoo"),
    ]

    builder = ResponseBuilder(
        prompt_helper,
        llm_predictor,
        mock_qa_prompt,
        mock_refine_prompt,
        texts=texts,
    )
    response = builder.get_response(query_str, mode=ResponseMode.COMPACT)
    assert str(response) == (
        "What is?:" "This\n\nis\n\na\n\nbar\n\n" "This\n\nis\n\na\n\ntest"
    )


@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(LLMPredictor, "__init__", return_value=None)
def test_tree_summarize_response(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    documents: List[Document],
) -> None:
    """Test give response."""
    # test response with ResponseMode.TREE_SUMMARIZE
    # NOTE: here we want to guarante that prompts have 0 extra tokens
    mock_refine_prompt_tmpl = "{query_str}{existing_answer}{context_msg}"
    mock_refine_prompt = RefinePrompt(mock_refine_prompt_tmpl)

    mock_qa_prompt_tmpl = "{context_str}{query_str}"
    mock_qa_prompt = QuestionAnswerPrompt(mock_qa_prompt_tmpl)

    # max input size is 12, prompt tokens is 2 (query_str)
    # --> 10 tokens for 2 chunks -->
    # 5 tokens per chunk, 1 is padding --> 4 tokens per chunk
    prompt_helper = PromptHelper(12, 0, 0, tokenizer=mock_tokenizer, separator="\n\n")

    # within tree_summarize, make sure that chunk size is 8
    llm_predictor = LLMPredictor()
    query_str = "What is?"
    texts = [
        TextChunk("This\n\nis\n\na\n\nbar"),
        TextChunk("This\n\nis\n\na\n\ntest"),
        TextChunk("This\n\nis\n\nanother\n\ntest"),
        TextChunk("This\n\nis\n\na\n\nfoo"),
    ]

    builder = ResponseBuilder(
        prompt_helper,
        llm_predictor,
        mock_qa_prompt,
        mock_refine_prompt,
        texts=texts,
    )
    response = builder.get_response(
        query_str, mode=ResponseMode.TREE_SUMMARIZE, num_children=2
    )
    # TODO: fix this output, the \n join appends unnecessary results at the end
    assert str(response) == (
        "What is?:This\n\nis\n\na\n\nbar\nThis\n" "This\n\nis\n\nanother\n\ntest\nThis"
    )
