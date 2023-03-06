"""Prompts for ChatGPT."""

from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
)
from gpt_index.prompts.prompts import RefinePrompt

CHAT_REFINE_PROMPT_TMPL_STRINGS = [
    (HumanMessagePromptTemplate, "{query_str}"),
    (AIMessagePromptTemplate, "{existing_answer}"),
    (
        HumanMessagePromptTemplate,
        "We have the opportunity to refine the above answer "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context, refine the original answer to better "
        "answer the question. "
        "If the context isn't useful, output the original answer again.",
    ),
]

CHAT_REFINE_PROMPT_LC = ChatPromptTemplate.from_strings(CHAT_REFINE_PROMPT_TMPL_STRINGS)

CHAT_REFINE_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_PROMPT_LC)


#     "The original question is as follows: {query_str}\n"
#     "We have provided an existing answer: {existing_answer}\n"
#     "We have the opportunity to refine the existing answer "
#     "(only if needed) with some more context below.\n"
#     "------------\n"
#     "{context_msg}\n"
#     "------------\n"
#     "Given the new context, refine the original answer to better "
#     "answer the question. "
#     "If the context isn't useful, return the original answer."
# )
# DEFAULT_REFINE_PROMPT = RefinePrompt(DEFAULT_REFINE_PROMPT_TMPL)
