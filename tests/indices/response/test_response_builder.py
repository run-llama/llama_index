"""Test response utils."""

import asyncio
from typing import List

from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.indices.response import ResponseMode, get_response_builder
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType
from llama_index.readers.schema.base import Document
from tests.indices.vector_store.mock_services import MockEmbedding
from tests.mock_utils.mock_prompts import MOCK_REFINE_PROMPT, MOCK_TEXT_QA_PROMPT


def mock_tokenizer(text: str) -> List[str]:
    """Mock tokenizer."""
    if text == "":
        return []
    tokens = text.split(" ")
    return tokens


def test_give_response(
    mock_service_context: ServiceContext,
    documents: List[Document],
) -> None:
    """Test give response."""
    prompt_helper = PromptHelper(
        context_window=DEFAULT_CONTEXT_WINDOW, num_output=DEFAULT_NUM_OUTPUTS
    )

    service_context = mock_service_context
    service_context.prompt_helper = prompt_helper
    query_str = "What is?"

    # test single line
    builder = get_response_builder(
        mode=ResponseMode.REFINE,
        service_context=service_context,
        text_qa_template=MOCK_TEXT_QA_PROMPT,
        refine_template=MOCK_REFINE_PROMPT,
    )
    response = builder.get_response(
        text_chunks=["This is a single line."], query_str=query_str
    )

    # test multiple lines
    response = builder.get_response(
        text_chunks=[documents[0].get_text()], query_str=query_str
    )
    expected_answer = (
        "What is?:"
        "Hello world.:"
        "This is a test.:"
        "This is another test.:"
        "This is a test v2."
    )
    assert str(response) == expected_answer


def test_compact_response(mock_service_context: ServiceContext) -> None:
    """Test give response."""
    # test response with ResponseMode.COMPACT
    # NOTE: here we want to guarante that prompts have 0 extra tokens
    mock_refine_prompt_tmpl = "{query_str}{existing_answer}{context_msg}"
    mock_refine_prompt = Prompt(mock_refine_prompt_tmpl, prompt_type=PromptType.REFINE)

    mock_qa_prompt_tmpl = "{context_str}{query_str}"
    mock_qa_prompt = Prompt(mock_qa_prompt_tmpl, prompt_type=PromptType.QUESTION_ANSWER)

    # max input size is 11, prompt is two tokens (the query) --> 9 tokens
    # --> padding is 1 --> 8 tokens
    prompt_helper = PromptHelper(
        max_input_size=11,
        num_output=0,
        max_chunk_overlap=0,
        tokenizer=mock_tokenizer,
        separator="\n\n",
        chunk_size_limit=4,
    )
    service_context = mock_service_context
    service_context.prompt_helper = prompt_helper
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
    builder = get_response_builder(
        service_context=service_context,
        text_qa_template=mock_qa_prompt,
        refine_template=mock_refine_prompt,
        mode=ResponseMode.COMPACT,
    )

    response = builder.get_response(text_chunks=texts, query_str=query_str)
    assert str(response) == "What is?:This:is:a:bar:This:is:a:test"


def test_tree_summarize_response(mock_service_context: ServiceContext) -> None:
    """Test give response."""
    # test response with ResponseMode.TREE_SUMMARIZE
    # NOTE: here we want to guarante that prompts have 0 extra tokens
    mock_refine_prompt_tmpl = "{query_str}{existing_answer}{context_msg}"
    mock_refine_prompt = Prompt(mock_refine_prompt_tmpl, prompt_type=PromptType.REFINE)

    mock_qa_prompt_tmpl = "{context_str}{query_str}"
    mock_qa_prompt = Prompt(mock_qa_prompt_tmpl, prompt_type=PromptType.QUESTION_ANSWER)

    # max input size is 20, prompt tokens is 2 (query_str)
    # --> 18 tokens for 2 chunks -->
    # 9 tokens per chunk, 5 is padding --> 4 tokens per chunk
    prompt_helper = PromptHelper(
        max_input_size=20,
        num_output=0,
        max_chunk_overlap=0,
        tokenizer=mock_tokenizer,
        separator="\n\n",
    )
    service_context = mock_service_context
    service_context.prompt_helper = prompt_helper

    # within tree_summarize, make sure that chunk size is 8
    query_str = "What is?"
    texts = [
        "This\n\nis\n\na\n\nbar",
        "This\n\nis\n\na\n\ntest",
        "This\n\nis\n\nanother\n\ntest",
        "This\n\nis\n\na\n\nfoo",
    ]

    builder = get_response_builder(
        mode=ResponseMode.TREE_SUMMARIZE,
        service_context=service_context,
        text_qa_template=mock_qa_prompt,
        refine_template=mock_refine_prompt,
    )

    response = builder.get_response(
        text_chunks=texts, query_str=query_str, num_children=2
    )
    # TODO: fix this output, the \n join appends unnecessary results at the end
    assert str(response) == "What is?:This:is:a:bar:This:is:another:test"


def test_accumulate_response(
    mock_service_context: ServiceContext,
    documents: List[Document],
) -> None:
    """Test accumulate response."""
    # test response with ResponseMode.ACCUMULATE
    # NOTE: here we want to guarante that prompts have 0 extra tokens
    mock_qa_prompt_tmpl = "{context_str}{query_str}"
    mock_qa_prompt = Prompt(mock_qa_prompt_tmpl, prompt_type=PromptType.QUESTION_ANSWER)

    # max input size is 11, prompt is two tokens (the query) --> 9 tokens
    # --> padding is 1 --> 8 tokens
    prompt_helper = PromptHelper(
        max_input_size=11,
        num_output=0,
        max_chunk_overlap=0,
        tokenizer=mock_tokenizer,
        separator="\n\n",
        chunk_size_limit=4,
    )
    service_context = mock_service_context
    service_context.prompt_helper = prompt_helper
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
    builder = get_response_builder(
        service_context=service_context,
        text_qa_template=mock_qa_prompt,
        mode=ResponseMode.ACCUMULATE,
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
    mock_service_context: ServiceContext,
    documents: List[Document],
) -> None:
    """Test accumulate response."""
    # test response with ResponseMode.ACCUMULATE
    # NOTE: here we want to guarante that prompts have 0 extra tokens
    mock_qa_prompt_tmpl = "{context_str}{query_str}"
    mock_qa_prompt = Prompt(mock_qa_prompt_tmpl, prompt_type=PromptType.QUESTION_ANSWER)

    # max input size is 11, prompt is two tokens (the query) --> 9 tokens
    # --> padding is 1 --> 8 tokens
    prompt_helper = PromptHelper(
        max_input_size=11,
        num_output=0,
        max_chunk_overlap=0,
        tokenizer=mock_tokenizer,
        separator="\n\n",
        chunk_size_limit=4,
    )
    service_context = mock_service_context
    service_context.prompt_helper = prompt_helper
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
    builder = get_response_builder(
        service_context=service_context,
        text_qa_template=mock_qa_prompt,
        mode=ResponseMode.ACCUMULATE,
        use_async=True,
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
    mock_service_context: ServiceContext,
    documents: List[Document],
) -> None:
    """Test accumulate response."""
    # test response with ResponseMode.ACCUMULATE
    # NOTE: here we want to guarante that prompts have 0 extra tokens
    mock_qa_prompt_tmpl = "{context_str}{query_str}"
    mock_qa_prompt = Prompt(mock_qa_prompt_tmpl, prompt_type=PromptType.QUESTION_ANSWER)

    # max input size is 11, prompt is two tokens (the query) --> 9 tokens
    # --> padding is 1 --> 8 tokens
    prompt_helper = PromptHelper(
        max_input_size=11,
        num_output=0,
        max_chunk_overlap=0,
        tokenizer=mock_tokenizer,
        separator="\n\n",
        chunk_size_limit=4,
    )
    service_context = mock_service_context
    service_context.prompt_helper = prompt_helper
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
    builder = get_response_builder(
        service_context=service_context,
        text_qa_template=mock_qa_prompt,
        mode=ResponseMode.ACCUMULATE,
    )

    response = asyncio.run(
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


def test_accumulate_compact_response(patch_llm_predictor: None) -> None:
    """Test accumulate response."""
    # test response with ResponseMode.ACCUMULATE
    # NOTE: here we want to guarante that prompts have 0 extra tokens
    mock_qa_prompt_tmpl = "{context_str}{query_str}"
    mock_qa_prompt = Prompt(mock_qa_prompt_tmpl, prompt_type=PromptType.QUESTION_ANSWER)

    # max input size is 11, prompt is two tokens (the query) --> 9 tokens
    # --> padding is 1 --> 8 tokens
    prompt_helper = PromptHelper(
        max_input_size=11,
        num_output=0,
        max_chunk_overlap=0,
        tokenizer=mock_tokenizer,
        separator="\n\n",
        chunk_size_limit=4,
    )
    service_context = ServiceContext.from_defaults(embed_model=MockEmbedding())
    service_context.prompt_helper = prompt_helper
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

    builder = get_response_builder(
        service_context=service_context,
        text_qa_template=mock_qa_prompt,
        mode=ResponseMode.COMPACT_ACCUMULATE,
    )

    response = builder.get_response(text_chunks=texts, query_str=query_str)
    expected = (
        "Response 1: What is?:This\n\nis\n\nbar\n\nThis"
        "\n---------------------\nResponse 2: What is?:is\n\nfoo"
    )
    assert str(response) == expected
