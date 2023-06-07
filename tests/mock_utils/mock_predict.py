"""Mock predict."""

import json
from typing import Any, Dict, Tuple

from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType
from llama_index.token_counter.utils import mock_extract_keywords_response


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


def _mock_single_select() -> str:
    """Mock single select."""
    return json.dumps(
        [
            {
                "choice": 1,
                "reason": "test",
            }
        ]
    )


def _mock_multi_select(prompt_args: Dict) -> str:
    """Mock single select."""
    answers = [
        {
            "choice": 1,
            "reason": "test",
        },
        {
            "choice": 2,
            "reason": "test",
        },
        {
            "choice": 3,
            "reason": "test",
        },
    ]
    max_outputs = prompt_args["max_outputs"]
    answers = answers[:max_outputs]

    return json.dumps(answers)


def _mock_sub_questions() -> str:
    """Mock sub questions."""
    json_str = json.dumps(
        [
            {
                "sub_question": "mock question for source_1",
                "tool_name": "source_1",
            }
        ],
        indent=4,
    )
    return f"```json\n{json_str}\n```"


def _mock_answer(prompt_args: Dict) -> str:
    """Mock answer."""
    return prompt_args["query_str"] + ":" + prompt_args["context_str"]


def _mock_refine(prompt_args: Dict) -> str:
    """Mock refine."""
    return prompt_args["existing_answer"] + ":" + prompt_args["context_msg"]


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


def _mock_input(prompt_args: Dict) -> str:
    """Mock input prompt."""
    return prompt_args["query_str"]


def _mock_decompose_query(prompt_args: Dict) -> str:
    """Mock decompose query."""
    return prompt_args["query_str"] + ":" + prompt_args["context_str"]


def _mock_pandas(prompt_args: Dict) -> str:
    """Mock pandas prompt."""
    query_str = prompt_args["query_str"]
    return f'df["{query_str}"].iloc[0]'


def _mock_choice_select(prompt_args: Dict) -> str:
    """Mock choice select prompt."""
    return "Doc: 1, Relevance: 5"


def _mock_sql_response_synthesis(prompt_args: Dict) -> str:
    """Mock sql response synthesis prompt."""
    return prompt_args["sql_response_str"]


def _mock_conversation(prompt_args: Dict) -> str:
    return prompt_args["history"] + ":" + prompt_args["message"]


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
    elif prompt.prompt_type == PromptType.SIMPLE_INPUT:
        response = _mock_input(full_prompt_args)
    elif prompt.prompt_type == PromptType.SINGLE_SELECT:
        response = _mock_single_select()
    elif prompt.prompt_type == PromptType.MULTI_SELECT:
        response = _mock_multi_select(full_prompt_args)
    elif prompt.prompt_type == PromptType.SUB_QUESTION:
        response = _mock_sub_questions()
    elif prompt.prompt_type == PromptType.PANDAS:
        response = _mock_pandas(full_prompt_args)
    elif prompt.prompt_type == PromptType.SQL_RESPONSE_SYNTHESIS:
        response = _mock_sql_response_synthesis(full_prompt_args)
    elif prompt.prompt_type == PromptType.DECOMPOSE:
        response = _mock_decompose_query(full_prompt_args)
    elif prompt.prompt_type == PromptType.CHOICE_SELECT:
        response = _mock_choice_select(full_prompt_args)
    elif prompt.prompt_type == PromptType.CONVERSATION:
        response = _mock_conversation(full_prompt_args)
    else:
        response = str(full_prompt_args)

    return response, formatted_prompt


def patch_llmpredictor_predict(
    self: Any, prompt: Prompt, **prompt_args: Any
) -> Tuple[str, str]:
    """Mock predict method of LLMPredictor.

    Depending on the prompt, return response.

    """
    return mock_llmpredictor_predict(prompt, **prompt_args)


async def patch_llmpredictor_apredict(
    self: Any, prompt: Prompt, **prompt_args: Any
) -> Tuple[str, str]:
    """Mock apredict method of LLMPredictor."""
    return patch_llmpredictor_predict(self, prompt, **prompt_args)


async def mock_llmpredictor_apredict(
    prompt: Prompt, **prompt_args: Any
) -> Tuple[str, str]:
    """Mock apredict method of LLMPredictor."""
    return mock_llmpredictor_predict(prompt, **prompt_args)


def mock_llmchain_predict(**full_prompt_args: Any) -> str:
    """Mock LLMChain predict with a generic response."""
    return "generic response from LLMChain.predict()"
