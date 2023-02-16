"""Mock predict."""

from typing import Any, Dict, Tuple

from gpt_index.prompts.base import Prompt
from gpt_index.prompts.prompt_type import PromptType
from gpt_index.token_counter.utils import mock_extract_keywords_response


def _mock_summary_predict(prompt_args: Dict) -> str:
    """Mock summary predict."""
    return prompt_args["context_str"]


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


def _mock_schema_extract(prompt_args: Dict) -> str:
    """Mock schema extract."""
    return prompt_args["text"]


def _mock_text_to_sql(prompt_args: Dict) -> str:
    """Mock text to sql."""
    # assume it's a select query
    tokens = prompt_args["query_str"].split(":")
    table_name = tokens[0]
    subtokens = tokens[1].split(",")
    return "SELECT " + ", ".join(subtokens) + f" FROM {table_name}"


def _mock_kg_triplet_extract(prompt_args: Dict) -> str:
    """Mock kg triplet extract."""
    return prompt_args["text"]


def mock_llmpredictor_predict(prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
    """Mock predict method of LLMPredictor.

    Depending on the prompt, return response.

    """
    formatted_prompt = prompt.format(**prompt_args)
    full_prompt_args = prompt.get_full_format_args(prompt_args)
    if prompt.prompt_type == PromptType.SUMMARY:
        response = _mock_summary_predict(full_prompt_args)
    elif prompt.prompt_type == PromptType.TREE_INSERT:
        response = _mock_insert_predict()
    elif prompt.prompt_type == PromptType.TREE_SELECT:
        response = _mock_query_select()
    elif prompt.prompt_type == PromptType.REFINE:
        response = _mock_refine(full_prompt_args)
    elif prompt.prompt_type == PromptType.QUESTION_ANSWER:
        response = _mock_answer(full_prompt_args)
    elif prompt.prompt_type == PromptType.KEYWORD_EXTRACT:
        response = _mock_keyword_extract(full_prompt_args)
    elif prompt.prompt_type == PromptType.QUERY_KEYWORD_EXTRACT:
        response = _mock_query_keyword_extract(full_prompt_args)
    elif prompt.prompt_type == PromptType.SCHEMA_EXTRACT:
        response = _mock_schema_extract(full_prompt_args)
    elif prompt.prompt_type == PromptType.TEXT_TO_SQL:
        response = _mock_text_to_sql(full_prompt_args)
    elif prompt.prompt_type == PromptType.KNOWLEDGE_TRIPLET_EXTRACT:
        response = _mock_kg_triplet_extract(full_prompt_args)
    else:
        raise ValueError("Invalid prompt to use with mocks.")

    return response, formatted_prompt


def mock_llmchain_predict(**full_prompt_args: Any) -> str:
    """Mock LLMChain predict with a generic response."""
    return "generic response from LLMChain.predict()"
