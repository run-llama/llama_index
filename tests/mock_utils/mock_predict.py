"""Mock predict."""

from typing import Any, Dict, Tuple

from gpt_index.prompts.base import Prompt
from tests.mock_utils.mock_prompts import (
    MOCK_INSERT_PROMPT,
    MOCK_KEYWORD_EXTRACT_PROMPT,
    MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
    MOCK_QUERY_PROMPT,
    MOCK_REFINE_PROMPT,
    MOCK_SUMMARY_PROMPT,
    MOCK_TEXT_QA_PROMPT,
)
from tests.mock_utils.mock_utils import mock_extract_keywords_response


def _mock_summary_predict(prompt_args: Dict) -> str:
    """Mock summary predict."""
    return prompt_args["text"]


def _mock_insert_predict() -> str:
    """Mock insert predict.

    Used in GPT tree index during insertion
    to select the next node.

    """
    return "ANSWER: 1"


def _mock_query_select() -> str:
    """Mock query predict.

    Used in GPT tree index during query traversal
    to select the next node.

    """
    return "ANSWER: 1"


def _mock_answer(prompt_args: Dict) -> str:
    """Mock answer."""
    return prompt_args["query_str"] + ":" + prompt_args["context_str"]


def _mock_refine(prompt_args: Dict) -> str:
    """Mock refine."""
    return prompt_args["existing_answer"]


def _mock_keyword_extract(prompt_args: Dict) -> str:
    """Mock keyword extract."""
    return mock_extract_keywords_response(prompt_args["text"])


def _mock_query_keyword_extract(prompt_args: Dict) -> str:
    """Mock query keyword extract."""
    return mock_extract_keywords_response(prompt_args["question"])


def mock_llmpredictor_predict(prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
    """Mock predict method of LLMPredictor.

    Depending on the prompt, return response.

    """
    formatted_prompt = prompt.format(**prompt_args)
    if prompt == MOCK_SUMMARY_PROMPT:
        response = _mock_summary_predict(prompt_args)
    elif prompt == MOCK_INSERT_PROMPT:
        response = _mock_insert_predict()
    elif prompt == MOCK_QUERY_PROMPT:
        response = _mock_query_select()
    elif prompt == MOCK_REFINE_PROMPT:
        response = _mock_refine(prompt_args)
    elif prompt == MOCK_TEXT_QA_PROMPT:
        response = _mock_answer(prompt_args)
    elif prompt == MOCK_KEYWORD_EXTRACT_PROMPT:
        response = _mock_keyword_extract(prompt_args)
    elif prompt == MOCK_QUERY_KEYWORD_EXTRACT_PROMPT:
        response = _mock_query_keyword_extract(prompt_args)
    else:
        raise ValueError("Invalid prompt to use with mocks.")

    return response, formatted_prompt


def mock_llmchain_predict(**full_prompt_args: Any) -> str:
    """Mock LLMChain predict with a generic response."""
    return "generic response from LLMChain.predict()"
