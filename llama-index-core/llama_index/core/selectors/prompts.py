from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

"""Single select prompt.

PromptTemplate to select one out of `num_choices` options provided in `context_list`,
given a query `query_str`.

Required template variables: `num_chunks`, `context_list`, `query_str`

"""
SingleSelectPrompt = PromptTemplate

"""Multiple select prompt.

PromptTemplate to select multiple candidates (up to `max_outputs`) out of `num_choices`
options provided in `context_list`, given a query `query_str`.

Required template variables: `num_chunks`, `context_list`, `query_str`,
    `max_outputs`
"""
MultiSelectPrompt = PromptTemplate


# single select
DEFAULT_SINGLE_SELECT_PROMPT_TMPL = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_choices}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return "
    "the choice that is most relevant to the question: '{query_str}'\n"
)


DEFAULT_SINGLE_SELECT_PROMPT = PromptTemplate(
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
    "(no more than {max_outputs}, but only select what is needed) that "
    "are most relevant to the question: '{query_str}'\n"
)


DEFAULT_MULTIPLE_SELECT_PROMPT = PromptTemplate(
    template=DEFAULT_MULTI_SELECT_PROMPT_TMPL, prompt_type=PromptType.MULTI_SELECT
)

# single pydantic select
DEFAULT_SINGLE_PYD_SELECT_PROMPT_TMPL = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_choices}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, generate "
    "the selection object and reason that is most relevant to the "
    "question: '{query_str}'\n"
)


# multiple pydantic select
DEFAULT_MULTI_PYD_SELECT_PROMPT_TMPL = (
    "Some choices are given below. It is provided in a numbered "
    "list (1 to {num_choices}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return the top choice(s) "
    "(no more than {max_outputs}, but only select what is needed) by generating "
    "the selection object and reasons that are most relevant to the "
    "question: '{query_str}'\n"
)
