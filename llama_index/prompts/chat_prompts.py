"""Prompts for ChatGPT."""

from llama_index.bridge.langchain import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from llama_index.prompts.prompts import (
    QuestionAnswerPrompt,
    SummaryPrompt,
    RefinePrompt,
    RefineTableContextPrompt,
)

# text qa prompt
TEXT_QA_SYSTEM_PROMPT = SystemMessagePromptTemplate.from_template(
    "You are an expert Q&A system that is trusted around the world.\n"
    "Always answer the question using the provided context information, "
    "and not prior knowledge.\n"
    "Some rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the context, ...' or "
    "'The context information ...' or anything along "
    "those lines."
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    HumanMessagePromptTemplate.from_template(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the question. If the answer is not in the context, inform "
        "the user that you can't answer the question.\n"
        "Question: {query_str}\n"
        "Answer: "
    ),
]

CHAT_TEXT_QA_PROMPT_LC = ChatPromptTemplate.from_messages(TEXT_QA_PROMPT_TMPL_MSGS)
CHAT_TEXT_QA_PROMPT = QuestionAnswerPrompt.from_langchain_prompt(CHAT_TEXT_QA_PROMPT_LC)


# Tree Summarize
TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    HumanMessagePromptTemplate.from_template(
        "Context information from multiple sources is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the information from multiple sources and not prior knowledge, "
        "answer the question. If the answer is not in the context, inform "
        "the user that you can't answer the question.\n"
        "Question: {query_str}\n"
        "Answer: "
    ),
]

CHAT_TREE_SUMMARIZE_PROMPT_LC = ChatPromptTemplate.from_messages(
    TREE_SUMMARIZE_PROMPT_TMPL_MSGS
)
CHAT_TREE_SUMMARIZE_PROMPT = SummaryPrompt.from_langchain_prompt(
    CHAT_TREE_SUMMARIZE_PROMPT_LC
)


# Refine Prompt
CHAT_REFINE_PROMPT_TMPL_MSGS = [
    HumanMessagePromptTemplate.from_template(
        "You are an expert Q&A system that stricly operates in two modes"
        "when refining existing answers:\n"
        "1. **Rewrite** an original answer using the new context.\n"
        "2. **Repeat** the original answer if the new context isn't useful.\n"
        "Never reference the original answer or context directly in your answer.\n"
        "When in doubt, just repeat the original answer."
        "New Context: {context_msg}\n"
        "Query: {query_str}\n"
        "Original Answer: {existing_answer}\n"
        "New Answer: "
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
