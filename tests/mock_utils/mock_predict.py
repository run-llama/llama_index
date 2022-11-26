"""Mock predict."""

from typing import Any, Dict, Optional, Tuple

from gpt_index.prompts.base import Prompt
from tests.mock_utils.mock_prompts import (
    MOCK_INSERT_PROMPT,
    MOCK_QUERY_PROMPT,
    MOCK_REFINE_PROMPT,
    MOCK_SUMMARY_PROMPT,
    MOCK_TEXT_QA_PROMPT,
)


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
    return prompt_args["query_str"] + "\n" + prompt_args["context_str"]


def _mock_refine(prompt_args: Dict) -> str:
    """Mock refine."""
    return prompt_args["existing_answer"]


def mock_openai_llm_predict(
    prompt: Prompt, llm_args_dict: Optional[Dict] = None, **prompt_args: Any
) -> Tuple[str, str]:
    """Mock OpenAI LLM predict.

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
    else:
        raise ValueError("Invalid prompt to use with mocks.")

    return response, formatted_prompt
