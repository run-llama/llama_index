"""Test response utils."""

from typing import List

from llama_index.core.async_utils import asyncio_run
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.schema import Document
from tests.mock_utils.mock_prompts import MOCK_REFINE_PROMPT, MOCK_TEXT_QA_PROMPT
from tests.mock_utils.mock_utils import mock_tokenizer


def test_give_response(
    documents: List[Document], patch_llm_predictor, patch_token_text_splitter
) -> None:
    """Test give response."""
    prompt_helper = PromptHelper(
        context_window=DEFAULT_CONTEXT_WINDOW, num_output=DEFAULT_NUM_OUTPUTS
    )
    query_str = "What is?"

    # test single line
    builder = get_response_synthesizer(
        response_mode=ResponseMode.REFINE,
        text_qa_template=MOCK_TEXT_QA_PROMPT,
        refine_template=MOCK_REFINE_PROMPT,
        prompt_helper=prompt_helper,
    )
    response = builder.get_response(
        text_chunks=["This is a single line."], query_str=query_str
    )

    # test multiple lines
    response = builder.get_response(
        text_chunks=[documents[0].get_content()], query_str=query_str
    )
    expected_answer = (
        "What is?:Hello world.:This is a test.:This is another test.:This is a test v2."
    )
    assert str(response) == expected_answer


def test_compact_response(patch_llm_predictor, patch_token_text_splitter) -> None:
    """Test give response."""
    # test response with ResponseMode.COMPACT
    # NOTE: here we want to guarantee that prompts have 0 extra tokens
    mock_refine_prompt_tmpl = "{query_str}{existing_answer}{context_msg}"
    mock_refine_prompt = PromptTemplate(
        mock_refine_prompt_tmpl, prompt_type=PromptType.REFINE
    )

    mock_qa_prompt_tmpl = "{context_str}{query_str}"
    mock_qa_prompt = PromptTemplate(
        mock_qa_prompt_tmpl, prompt_type=PromptType.QUESTION_ANSWER
    )

    # max input size is 11, prompt is two tokens (the query) --> 9 tokens
    # --> padding is 1 --> 8 tokens
    prompt_helper = PromptHelper(
        context_window=11,
        num_output=0,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        separator="\n\n",
        chunk_size_limit=4,
    )
    cur_chunk_size = prompt_helper._get_available_chunk_size(
        mock_qa_prompt, 1, padding=1
    )
    # outside of compact, assert that chunk size is 4
    assert cur_chunk_size == 4

    # within compact, make sure that chunk size is 8
    query_str = "What is?"
    texts = [
        "This\n\nis\n\na\n\nbar",
        "This\n\nis\n\na\n\ntest",
    ]
    builder = get_response_synthesizer(
        text_qa_template=mock_qa_prompt,
        refine_template=mock_refine_prompt,
        response_mode=ResponseMode.COMPACT,
        prompt_helper=prompt_helper,
    )

    response = builder.get_response(text_chunks=texts, query_str=query_str)
    assert str(response) == "What is?:This:is:a:bar:This:is:a:test"


def test_accumulate_response(patch_llm_predictor, patch_token_text_splitter) -> None:
    """Test accumulate response."""
    # test response with ResponseMode.ACCUMULATE
    # NOTE: here we want to guarantee that prompts have 0 extra tokens
    mock_qa_prompt_tmpl = "{context_str}{query_str}"
    mock_qa_prompt = PromptTemplate(
        mock_qa_prompt_tmpl, prompt_type=PromptType.QUESTION_ANSWER
    )

    # max input size is 11, prompt is two tokens (the query) --> 9 tokens
    # --> padding is 1 --> 8 tokens
    prompt_helper = PromptHelper(
        context_window=11,
        num_output=0,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        separator="\n\n",
        chunk_size_limit=4,
    )
    cur_chunk_size = prompt_helper._get_available_chunk_size(
        mock_qa_prompt, 1, padding=1
    )
    # outside of compact, assert that chunk size is 4
    assert cur_chunk_size == 4

    # within compact, make sure that chunk size is 8
    query_str = "What is?"
    texts = [
        "This\nis\nbar",
        "This\nis\nfoo",
    ]
    builder = get_response_synthesizer(
        text_qa_template=mock_qa_prompt,
        response_mode=ResponseMode.ACCUMULATE,
        prompt_helper=prompt_helper,
    )

    response = builder.get_response(text_chunks=texts, query_str=query_str)
    expected = (
        "Response 1: What is?:This\n"
        "---------------------\n"
        "Response 2: What is?:is\n"
        "---------------------\n"
        "Response 3: What is?:bar\n"
        "---------------------\n"
        "Response 4: What is?:This\n"
        "---------------------\n"
        "Response 5: What is?:is\n"
        "---------------------\n"
        "Response 6: What is?:foo"
    )
    assert str(response) == expected


def test_accumulate_response_async(
    patch_llm_predictor, patch_token_text_splitter
) -> None:
    """Test accumulate response."""
    # test response with ResponseMode.ACCUMULATE
    # NOTE: here we want to guarantee that prompts have 0 extra tokens
    mock_qa_prompt_tmpl = "{context_str}{query_str}"
    mock_qa_prompt = PromptTemplate(
        mock_qa_prompt_tmpl, prompt_type=PromptType.QUESTION_ANSWER
    )

    # max input size is 11, prompt is two tokens (the query) --> 9 tokens
    # --> padding is 1 --> 8 tokens
    prompt_helper = PromptHelper(
        context_window=11,
        num_output=0,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        separator="\n\n",
        chunk_size_limit=4,
    )
    cur_chunk_size = prompt_helper._get_available_chunk_size(
        mock_qa_prompt, 1, padding=1
    )
    # outside of compact, assert that chunk size is 4
    assert cur_chunk_size == 4

    # within compact, make sure that chunk size is 8
    query_str = "What is?"
    texts = [
        "This\nis\nbar",
        "This\nis\nfoo",
    ]
    builder = get_response_synthesizer(
        text_qa_template=mock_qa_prompt,
        response_mode=ResponseMode.ACCUMULATE,
        use_async=True,
        prompt_helper=prompt_helper,
    )

    response = builder.get_response(text_chunks=texts, query_str=query_str)
    expected = (
        "Response 1: What is?:This\n"
        "---------------------\n"
        "Response 2: What is?:is\n"
        "---------------------\n"
        "Response 3: What is?:bar\n"
        "---------------------\n"
        "Response 4: What is?:This\n"
        "---------------------\n"
        "Response 5: What is?:is\n"
        "---------------------\n"
        "Response 6: What is?:foo"
    )
    assert str(response) == expected


def test_accumulate_response_aget(
    patch_llm_predictor, patch_token_text_splitter
) -> None:
    """Test accumulate response."""
    # test response with ResponseMode.ACCUMULATE
    # NOTE: here we want to guarantee that prompts have 0 extra tokens
    mock_qa_prompt_tmpl = "{context_str}{query_str}"
    mock_qa_prompt = PromptTemplate(
        mock_qa_prompt_tmpl, prompt_type=PromptType.QUESTION_ANSWER
    )

    # max input size is 11, prompt is two tokens (the query) --> 9 tokens
    # --> padding is 1 --> 8 tokens
    prompt_helper = PromptHelper(
        context_window=11,
        num_output=0,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        separator="\n\n",
        chunk_size_limit=4,
    )
    cur_chunk_size = prompt_helper._get_available_chunk_size(
        mock_qa_prompt, 1, padding=1
    )
    # outside of compact, assert that chunk size is 4
    assert cur_chunk_size == 4

    # within compact, make sure that chunk size is 8
    query_str = "What is?"
    texts = [
        "This\nis\nbar",
        "This\nis\nfoo",
    ]
    builder = get_response_synthesizer(
        text_qa_template=mock_qa_prompt,
        response_mode=ResponseMode.ACCUMULATE,
        prompt_helper=prompt_helper,
    )

    response = asyncio_run(
        builder.aget_response(
            text_chunks=texts,
            query_str=query_str,
            separator="\nWHATEVER~~~~~~\n",
        )
    )
    expected = (
        "Response 1: What is?:This\n"
        "WHATEVER~~~~~~\n"
        "Response 2: What is?:is\n"
        "WHATEVER~~~~~~\n"
        "Response 3: What is?:bar\n"
        "WHATEVER~~~~~~\n"
        "Response 4: What is?:This\n"
        "WHATEVER~~~~~~\n"
        "Response 5: What is?:is\n"
        "WHATEVER~~~~~~\n"
        "Response 6: What is?:foo"
    )
    assert str(response) == expected


def test_accumulate_compact_response(patch_llm_predictor):
    """Test accumulate response."""
    # test response with ResponseMode.ACCUMULATE
    # NOTE: here we want to guarantee that prompts have 0 extra tokens
    mock_qa_prompt_tmpl = "{context_str}{query_str}"
    mock_qa_prompt = PromptTemplate(
        mock_qa_prompt_tmpl, prompt_type=PromptType.QUESTION_ANSWER
    )

    # max input size is 11, prompt is two tokens (the query) --> 9 tokens
    # --> padding is 1 --> 8 tokens
    prompt_helper = PromptHelper(
        context_window=11,
        num_output=0,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        separator="\n\n",
        chunk_size_limit=4,
    )
    cur_chunk_size = prompt_helper._get_available_chunk_size(
        mock_qa_prompt, 1, padding=1
    )
    # outside of compact, assert that chunk size is 4
    assert cur_chunk_size == 4

    # within compact, make sure that chunk size is 8
    query_str = "What is?"
    texts = [
        "This",
        "is",
        "bar",
        "This",
        "is",
        "foo",
    ]
    compacted_chunks = prompt_helper.repack(mock_qa_prompt, texts)
    assert compacted_chunks == ["This\n\nis\n\nbar\n\nThis", "is\n\nfoo"]

    builder = get_response_synthesizer(
        text_qa_template=mock_qa_prompt,
        response_mode=ResponseMode.COMPACT_ACCUMULATE,
        prompt_helper=prompt_helper,
    )

    response = builder.get_response(text_chunks=texts, query_str=query_str)
    expected = (
        "Response 1: What is?:This\n\nis\n\nbar\n\nThis"
        "\n---------------------\nResponse 2: What is?:is\n\nfoo"
    )
    assert str(response) == expected
