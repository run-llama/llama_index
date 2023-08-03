"""Prompts for ChatGPT."""

from llama_index.bridge.langchain import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from llama_index.prompts.prompts import RefinePrompt, RefineTableContextPrompt

# text qa prompt
TEXT_QA_PROMPT_TMPL_MSGS = [
    SystemMessagePromptTemplate.from_template(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the question using the provided context information.\n"
        "Some rules to follow:\n"
        "1. Never directly reference the given context in your answer.\n"
        "2. Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along "
        "those lines."
    ),
    HumanMessagePromptTemplate.from_template(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the question: {query_str}\n"
    ),
]

CHAT_TEXT_QA_PROMPT_LC = ChatPromptTemplate.from_messages(TEXT_QA_PROMPT_TMPL_MSGS)
CHAT_TEXT_QA_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_TEXT_QA_PROMPT_LC)

# Refine Prompt
CHAT_REFINE_PROMPT_TMPL_MSGS = [
    SystemMessagePromptTemplate.from_template(
        "You are an expert Q&A system that follows strict rules:\n"
        "1. **Rewrite** an original answer using new context information\n"
        "2. **Repeat** the original answer if the context isn't useful\n"
        "3. Never mention or reference the orginal answer."
    ),
    HumanMessagePromptTemplate.from_template(
        "We have the opportunity to refine an original answer "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context, refine the original answer to better "
        "answer the question: {query_str}. "
        "If the context isn't useful, output the original answer again.\n"
        "Original Answer: {existing_answer}"
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
