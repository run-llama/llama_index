"""Response refine functions."""

from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.utils import truncate_text
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.default_prompts import (
    DEFAULT_REFINE_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt


def refine_response(
    prompt_helper: PromptHelper,
    llm_predictor: LLMPredictor,
    response: str,
    query_str: str,
    text_chunk: str,
    refine_template: RefinePrompt = DEFAULT_REFINE_PROMPT,
    verbose: bool = False,
) -> str:
    """Refine response."""
    fmt_text_chunk = truncate_text(text_chunk, 50)
    if verbose:
        print(f"> Refine context: {fmt_text_chunk}")
    refine_text_splitter = prompt_helper.get_text_splitter_given_prompt(
        refine_template, 1
    )
    text_chunks = refine_text_splitter.split_text(text_chunk)
    for cur_text_chunk in text_chunks:
        response, _ = llm_predictor.predict(
            refine_template,
            query_str=query_str,
            existing_answer=response,
            context_msg=cur_text_chunk,
        )
        if verbose:
            print(f"> Refined response: {response}")
    return response


def give_response(
    prompt_helper: PromptHelper,
    llm_predictor: LLMPredictor,
    query_str: str,
    text_chunk: str,
    text_qa_template: QuestionAnswerPrompt = DEFAULT_TEXT_QA_PROMPT,
    refine_template: RefinePrompt = DEFAULT_REFINE_PROMPT,
    verbose: bool = False,
) -> str:
    """Give response given a query and a corresponding text chunk."""
    qa_text_splitter = prompt_helper.get_text_splitter_given_prompt(text_qa_template, 1)
    text_chunks = qa_text_splitter.split_text(text_chunk)
    response = None
    for cur_text_chunk in text_chunks:
        if response is None:
            response, _ = llm_predictor.predict(
                text_qa_template, query_str=query_str, context_str=cur_text_chunk
            )
            if verbose:
                print(f"> Initial response: {response}")
        else:
            response = refine_response(
                prompt_helper,
                llm_predictor,
                response,
                query_str,
                cur_text_chunk,
                refine_template=refine_template,
                verbose=verbose,
            )
    return response or ""
