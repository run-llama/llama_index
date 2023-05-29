"""Subclasses from base prompt."""

from llama_index.prompts.base import Prompt

# deprecated, kept for backward compatibility

"""Summary prompt.

Prompt to summarize the provided `context_str`.

Required template variables: `context_str`
"""
SummaryPrompt = Prompt

"""Tree Insert prompt.

Prompt to insert a new chunk of text `new_chunk_text` into the tree index.
More specifically, this prompt has the LLM select the relevant candidate
child node to continue tree traversal.

Required template variables: `num_chunks`, `context_list`, `new_chunk_text`
"""
TreeInsertPrompt = Prompt

"""Tree select prompt.

Prompt to select a candidate child node out of all child nodes
provided in `context_list`, given a query `query_str`. `num_chunks` is
the number of child nodes in `context_list`.

Required template variables: `num_chunks`, `context_list`, `query_str`

"""
TreeSelectPrompt = Prompt

"""Tree select multiple prompt.

Prompt to select multiple candidate child nodes out of all
child nodes provided in `context_list`, given a query `query_str`.
`branching_factor` refers to the number of child nodes to select, and
`num_chunks` is the number of child nodes in `context_list`.

Required template variables: `num_chunks`, `context_list`, `query_str`,
    `branching_factor`
"""
TreeSelectMultiplePrompt = Prompt

"""Refine prompt.

Prompt to refine an existing answer `existing_answer` given a context `context_msg`,
and a query `query_str`.

Required template variables: `query_str`, `existing_answer`, `context_msg`
"""
RefinePrompt = Prompt

"""Question Answer prompt.

Prompt to answer a question `query_str` given a context `context_str`.

Required template variables: `context_str`, `query_str`
"""
QuestionAnswerPrompt = Prompt

"""Keyword extract prompt.

Prompt to extract keywords from a text `text` with a maximum of
`max_keywords` keywords.

Required template variables: `text`, `max_keywords`
"""
KeywordExtractPrompt = Prompt

"""Query keyword extract prompt.

Prompt to extract keywords from a query `query_str` with a maximum
of `max_keywords` keywords.

Required template variables: `query_str`, `max_keywords`
"""
QueryKeywordExtractPrompt = Prompt

"""Schema extract prompt.

Prompt to extract schema from unstructured text `text`.

Required template variables: `text`, `schema`
"""
SchemaExtractPrompt = Prompt

"""Text to SQL prompt.

Prompt to translate a natural language query into SQL in the dialect
`dialect` given a schema `schema`.

Required template variables: `query_str`, `schema`, `dialect`
"""
TextToSQLPrompt = Prompt
"""Table context prompt.

Prompt to generate a table context given a table schema `schema`,
as well as unstructured text context `context_str`, and
a task `query_str`.
This includes both a high-level description of the table
as well as a description of each column in the table.
"""
TableContextPrompt = Prompt

"""Refine Table context prompt.

Prompt to refine a table context given a table schema `schema`,
as well as unstructured text context `context_msg`, and
a task `query_str`.
This includes both a high-level description of the table
as well as a description of each column in the table.

"""
RefineTableContextPrompt = Prompt

"""Define the knowledge graph triplet extraction prompt."""
KnowledgeGraphPrompt = Prompt

"""Simple Input prompt.

Required template variables: `query_str`.
"""
SimpleInputPrompt = Prompt

"""Pandas prompt. Convert query to python code.

Required template variables: `query_str`, `df_str`, `instruction_str`.
"""
PandasPrompt = Prompt
