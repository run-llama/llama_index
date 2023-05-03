from typing import List
from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType


class SingleSelectPrompt(Prompt):
    """Single select prompt.

    Prompt to select one out of `num_choices` options provided in `context_list`,
    given a query `query_str`.

    Required template variables: `num_chunks`, `context_list`, `query_str`

    Args:
        template (str): Template for the prompt.
        **prompt_kwargs: Keyword arguments for the prompt.

    """

    prompt_type: PromptType = PromptType.SINGLE_SELECT
    input_variables: List[str] = ["num_choices", "context_list", "query_str"]


# single select
DEFAULT_SINGLE_SELECT_PROMPT_TMPL = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_choices}),"
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return "
    "the choice that is most relevant to the question: '{query_str}'\n"
)


DEFAULT_SINGLE_SELECT_PROMPT = SingleSelectPrompt(
    template=DEFAULT_SINGLE_SELECT_PROMPT_TMPL
)


class MultiSelectPrompt(Prompt):
    """Multiple select prompt.

    Prompt to select multiple candidates (up to `max_outputs`) out of `num_choices`
    options provided in `context_list`, given a query `query_str`.

    Required template variables: `num_chunks`, `context_list`, `query_str`,
        `max_outputs`

    Args:
        template (str): Template for the prompt.
        **prompt_kwargs: Keyword arguments for the prompt.

    """

    prompt_type: PromptType = PromptType.MULTI_SELECT
    input_variables: List[str] = [
        "num_choices",
        "context_list",
        "query_str",
        "max_outputs",
    ]


# multiple select
DEFAULT_MULTI_SELECT_PROMPT_TMPL = (
    "Some choices are given below. It is provided in a numbered "
    "list (1 to {num_choices}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return the top choices "
    "(no more than {max_outputs}, ranked by most relevant to least) that "
    "are most relevant to the question: '{query_str}'\n"
)


DEFAULT_MULTIPLE_SELECT_PROMPT = MultiSelectPrompt(
    template=DEFAULT_MULTI_SELECT_PROMPT_TMPL
)
