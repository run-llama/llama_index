from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType

"""Single select prompt.

Prompt to select one out of `num_choices` options provided in `context_list`,
given a query `query_str`.

Required template variables: `num_chunks`, `context_list`, `query_str`

"""
SingleSelectPrompt = Prompt

"""Multiple select prompt.

Prompt to select multiple candidates (up to `max_outputs`) out of `num_choices`
options provided in `context_list`, given a query `query_str`.

Required template variables: `num_chunks`, `context_list`, `query_str`,
    `max_outputs`
"""
MultiSelectPrompt = Prompt


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


DEFAULT_SINGLE_SELECT_PROMPT = Prompt(
    template=DEFAULT_SINGLE_SELECT_PROMPT_TMPL, prompt_type=PromptType.SINGLE_SELECT
)


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


DEFAULT_MULTIPLE_SELECT_PROMPT = Prompt(
    template=DEFAULT_MULTI_SELECT_PROMPT_TMPL, prompt_type=PromptType.MULTI_SELECT
)
