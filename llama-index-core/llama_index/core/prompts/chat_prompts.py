"""Prompts for ChatGPT."""

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.prompts.base import ChatPromptTemplate

# text qa prompt
TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information, "
        "and not prior knowledge.\n"
        "Some rules to follow:\n"
        "1. Never directly reference the given context in your answer.\n"
        "2. Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along "
        "those lines."
    ),
    role=MessageRole.SYSTEM,
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

# Tree Summarize
TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information from multiple sources is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the information from multiple sources and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TREE_SUMMARIZE_PROMPT = ChatPromptTemplate(
    message_templates=TREE_SUMMARIZE_PROMPT_TMPL_MSGS
)


# Refine Prompt
CHAT_REFINE_PROMPT_TMPL_MSGS = [
    ChatMessage(
        content=(
            "You are an expert Q&A system that strictly operates in two modes "
            "when refining existing answers:\n"
            "1. **Rewrite** an original answer using the new context.\n"
            "2. **Repeat** the original answer if the new context isn't useful.\n"
            "Never reference the original answer or context directly in your answer.\n"
            "When in doubt, just repeat the original answer.\n"
            "New Context: {context_msg}\n"
            "Query: {query_str}\n"
            "Original Answer: {existing_answer}\n"
            "New Answer: "
        ),
        role=MessageRole.USER,
    )
]


CHAT_REFINE_PROMPT = ChatPromptTemplate(message_templates=CHAT_REFINE_PROMPT_TMPL_MSGS)


# Table Context Refine Prompt
CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS = [
    ChatMessage(content="{query_str}", role=MessageRole.USER),
    ChatMessage(content="{existing_answer}", role=MessageRole.ASSISTANT),
    ChatMessage(
        content=(
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
        role=MessageRole.USER,
    ),
]
CHAT_REFINE_TABLE_CONTEXT_PROMPT = ChatPromptTemplate(
    message_templates=CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS
)


# Choice Select
CHAT_CHOICE_SELECT_PROMPT = RichPromptTemplate("""
{% chat role="user" %}
A list of documents is shown below. Each document has a number next to it along \
with a summary of the document. A question is also provided.

Respond with the numbers of the documents you should consult to answer the question, \
in order of relevance, as well as the relevance score. \
The relevance score is a number from 1-10 based on how relevant you think the document \
is to the question."

Do not include any documents that are not relevant to the question.

Example format:

Document 1:
<summary of document 1>

Document 2:
<summary of document 2>

...

Document 10:
<summary of document 10>

Question: <question>

Answer:
Doc: 9, Relevance: 7
Doc: 3, Relevance: 4
Doc: 7, Relevance: 3

Let's try this now:

{% for message in context_messages %}
Document {{ loop.index }}:
{% for block in message.blocks %}
{% if block.block_type == 'text' %}
{{ block.text }}
{% elif block.block_type == 'image' %}
{{ block.inline_url() | image }}
{% elif block.block_type == 'audio' %}
{{ block.inline_url() | audio }}
{% elif block.block_type == 'video' %}
{{ block.inline_url() | video }}
{% endif %}

{% endfor %}
{% endfor %}

Question: {{ query_str }}

Answer:
{% endchat %}
""")
