"""Prompts for ChatGPT."""

from llama_index.bridge.langchain import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from llama_index.prompts.prompts import RefinePrompt, RefineTableContextPrompt

# Refine Prompt
CHAT_REFINE_PROMPT_TMPL_MSGS = [
    HumanMessagePromptTemplate.from_template(
        "We have the opportunity to refine the above answer "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context, refine the original answer to better "
        "answer the question: {query_str}. "
        "If the context isn't useful, output the original answer again.\n"
        "------------\n"
        "{existing_answer}\n"
        "------------"
    ),
]


CHAT_REFINE_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_PROMPT_TMPL_MSGS)
CHAT_REFINE_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_PROMPT_LC)


# Table Context Refine Prompt
CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS = [
    HumanMessagePromptTemplate.from_template("{query_str}"),
    AIMessagePromptTemplate.from_template("{existing_answer}"),
    HumanMessagePromptTemplate.from_template(
        "We have provided a table schema below. "
        "---------------------\n"
        "{schema}\n"
        "---------------------\n"
        "We have also provided some context information below. "
        "{context_msg}\n"
        "---------------------\n"
        "Given the context information and the table schema, "
        "refine the original answer to better "
        "answer the question. "
        "If the context isn't useful, return the original answer."
    ),
]
CHAT_REFINE_TABLE_CONTEXT_PROMPT_LC = ChatPromptTemplate.from_messages(
    CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS
)
CHAT_REFINE_TABLE_CONTEXT_PROMPT = RefineTableContextPrompt.from_langchain_prompt(
    CHAT_REFINE_TABLE_CONTEXT_PROMPT_LC
)
