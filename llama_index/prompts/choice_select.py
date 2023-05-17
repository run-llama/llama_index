"""Default choice select prompt."""

from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType
from typing import List


class ChoiceSelectPrompt(Prompt):
    """Choice Select prompt.

    Prompt to return relevant context from `context_str` given a query `query_str`.

    Required template variables: `context_str`, `query_str`

    Args:
        template (str): Template for the prompt.
        **prompt_kwargs: Keyword arguments for the prompt.

    """

    prompt_type: PromptType = PromptType.CUSTOM
    input_variables: List[str] = ["context_str", "query_str"]


DEFAULT_CHOICE_SELECT_PROMPT_TMPL = (
    "A list of documents is shown below. Each document has a number next to it along "
    "with a summary of the document. A question is also provided. \n"
    "Respond with the numbers of the documents "
    "you should consult to answer the question, in order of relevance, as well \n"
    "as the relevance score. The relevance score is a number from 1-10 based on "
    "how relevant you think the document is to the question.\n"
    "Do not include any documents that are not relevant to the question. \n"
    "Example format: \n"
    "Document 1:\n<summary of document 1>\n\n"
    "Document 2:\n<summary of document 2>\n\n"
    "...\n\n"
    "Document 10:\n<summary of document 10>\n\n"
    "Question: <question>\n"
    "Answer:\n"
    "Doc: 9, Relevance: 7\n"
    "Doc: 3, Relevance: 4\n"
    "Doc: 7, Relevance: 3\n\n"
    "Let's try this now: \n\n"
    "{context_str}\n"
    "Question: {query_str}\n"
    "Answer:\n"
)
DEFAULT_CHOICE_SELECT_PROMPT = ChoiceSelectPrompt(DEFAULT_CHOICE_SELECT_PROMPT_TMPL)
