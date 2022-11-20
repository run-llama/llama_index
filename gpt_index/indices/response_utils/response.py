"""Response refine functions."""

from gpt_index.constants import MAX_CHUNK_OVERLAP, MAX_CHUNK_SIZE, NUM_OUTPUTS
from gpt_index.indices.utils import get_chunk_size_given_prompt, truncate_text
from gpt_index.langchain_helpers.chain_wrapper import openai_llm_predict
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import (
    DEFAULT_REFINE_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)


def refine_response(
    response: str,
    query_str: str,
    text_chunk: str,
    refine_template: Prompt = DEFAULT_REFINE_PROMPT,
    verbose: bool = False,
) -> str:
    """Refine response."""
    fmt_text_chunk = truncate_text(text_chunk, 50)
    if verbose:
        print(f"> Refine context: {fmt_text_chunk}")
    empty_refine_template = refine_template.format(
        query_str=query_str,
        existing_answer=response,
        context_msg="",
    )
    refine_chunk_size = get_chunk_size_given_prompt(
        empty_refine_template, MAX_CHUNK_SIZE, 1, NUM_OUTPUTS
    )
    refine_text_splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=refine_chunk_size,
        chunk_overlap=MAX_CHUNK_OVERLAP,
    )
    text_chunks = refine_text_splitter.split_text(text_chunk)
    for text_chunk in text_chunks:
        response, _ = openai_llm_predict(
            refine_template,
            query_str=query_str,
            existing_answer=response,
            context_msg=text_chunk,
        )
        if verbose:
            print(f"> Refined response: {response}")
    return response


def give_response(
    query_str: str,
    text_chunk: str,
    text_qa_template: Prompt = DEFAULT_TEXT_QA_PROMPT,
    refine_template: Prompt = DEFAULT_REFINE_PROMPT,
    verbose: bool = False,
) -> str:
    """Give response given a query and a corresponding text chunk."""
    empty_text_qa_template = text_qa_template.format(
        query_str=query_str,
        context_str="",
    )
    qa_chunk_size = get_chunk_size_given_prompt(
        empty_text_qa_template, MAX_CHUNK_SIZE, 1, NUM_OUTPUTS
    )
    qa_text_splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=qa_chunk_size,
        chunk_overlap=MAX_CHUNK_OVERLAP,
    )
    text_chunks = qa_text_splitter.split_text(text_chunk)
    response = None
    for text_chunk in text_chunks:
        if response is None:
            response, _ = openai_llm_predict(
                text_qa_template, query_str=query_str, context_str=text_chunk
            )
            if verbose:
                print(f"> Initial response: {response}")
        else:
            response = refine_response(
                response,
                query_str,
                text_chunk,
                refine_template=refine_template,
                verbose=verbose,
            )
    return response or ""
