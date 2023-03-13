"""Init params."""

from typing import List

from gpt_index.prompts.base import Prompt
from gpt_index.prompts.prompt_type import PromptType


# class QueryCombineDonePrompt(Prompt):
#     """Query combine done prompt.

#     Prompt to decide whether followup questions are needed over a knowledge source,
#     given the existing context + previous reasoning (the previous steps).

#     Required template variables: `context_str`, `query_str`, `prev_reasoning`

#     Args:
#         template (str): Template for the prompt.
#         **prompt_kwargs: Keyword arguments for the prompt.

#     """

#     # TODO: specify a better prompt type
#     prompt_type: PromptType = PromptType.CUSTOM
#     input_variables: List[str] = ["context_str", "query_str", "prev_reasoning"]


# DEFAULT_IF_DONE_TMPL = (
#     "We wish to decide if followup questions are needed over a knowledge source, "
#     "given a high-level question. "
#     "Please respond with a yes or no.\n\n"
#     "Examples:\n\n"
#     "High-level question: How many Grand Slam titles does the winner of the 2020 Australian "
#     "Open have?\n"
#     "Knowledge Source Context: Provides information about the winners of the 2020 "
#     "Australian Open\n"
#     "Previous reasoning: None."
#     "Followup questions needed over knowledge source: Yes. "
#     "Current query: How many Grand Slam titles does the winner of the 2020 Australian "
#     "Open have?\n"
#     "Knowledge Source Context: Provides information about the winners of the 2020 "
#     "Australian Open\n"
#     "Previous reasoning:\n"
#     "- Who was the winner of the 2020 Australian Open? \n"
#     "- The winner of the 2020 Australian Open was Novak Djokovic.\n"
#     "Followup questions neeeded over knowledge source: No\n "
#     "High-level question: {query_str}\n"
#     "Knowledge Source Context: {knowledge_source}\n"
#     "Previous reasoning: \n{prev_reasoning}\n"
#     "Followup questions needed: "
# )

# DEFAULT_IF_DONE_PROMPT = QueryCombineDonePrompt(DEFAULT_IF_DONE_TMPL)
