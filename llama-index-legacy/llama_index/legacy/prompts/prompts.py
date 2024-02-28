"""Subclasses from base prompt."""

from llama_index.legacy.prompts.base import PromptTemplate

# deprecated, kept for backward compatibility

"""Summary prompt.

PromptTemplate to summarize the provided `context_str`.

Required template variables: `context_str`
"""
SummaryPrompt = PromptTemplate

"""Tree Insert prompt.

PromptTemplate to insert a new chunk of text `new_chunk_text` into the tree index.
More specifically, this prompt has the LLM select the relevant candidate
child node to continue tree traversal.

Required template variables: `num_chunks`, `context_list`, `new_chunk_text`
"""
TreeInsertPrompt = PromptTemplate

"""Tree select prompt.

PromptTemplate to select a candidate child node out of all child nodes
provided in `context_list`, given a query `query_str`. `num_chunks` is
the number of child nodes in `context_list`.

Required template variables: `num_chunks`, `context_list`, `query_str`

"""
TreeSelectPrompt = PromptTemplate

"""Tree select multiple prompt.

PromptTemplate to select multiple candidate child nodes out of all
child nodes provided in `context_list`, given a query `query_str`.
`branching_factor` refers to the number of child nodes to select, and
`num_chunks` is the number of child nodes in `context_list`.

Required template variables: `num_chunks`, `context_list`, `query_str`,
    `branching_factor`
"""
TreeSelectMultiplePrompt = PromptTemplate

"""Refine prompt.

PromptTemplate to refine an existing answer `existing_answer`
given a context `context_msg`, and a query `query_str`.

Required template variables: `query_str`, `existing_answer`, `context_msg`
"""
RefinePrompt = PromptTemplate

"""Question Answer prompt.

PromptTemplate to answer a question `query_str` given a context `context_str`.

Required template variables: `context_str`, `query_str`
"""
QuestionAnswerPrompt = PromptTemplate

"""Keyword extract prompt.

PromptTemplate to extract keywords from a text `text` with a maximum of
`max_keywords` keywords.

Required template variables: `text`, `max_keywords`
"""
KeywordExtractPrompt = PromptTemplate

"""Query keyword extract prompt.

PromptTemplate to extract keywords from a query `query_str` with a maximum
of `max_keywords` keywords.

Required template variables: `query_str`, `max_keywords`
"""
QueryKeywordExtractPrompt = PromptTemplate

"""Schema extract prompt.

PromptTemplate to extract schema from unstructured text `text`.

Required template variables: `text`, `schema`
"""
SchemaExtractPrompt = PromptTemplate

"""Text to SQL prompt.

PromptTemplate to translate a natural language query into SQL in the dialect
`dialect` given a schema `schema`.

Required template variables: `query_str`, `schema`, `dialect`
"""
TextToSQLPrompt = PromptTemplate
"""Table context prompt.

PromptTemplate to generate a table context given a table schema `schema`,
as well as unstructured text context `context_str`, and
a task `query_str`.
This includes both a high-level description of the table
as well as a description of each column in the table.
"""
TableContextPrompt = PromptTemplate

"""Refine Table context prompt.

PromptTemplate to refine a table context given a table schema `schema`,
as well as unstructured text context `context_msg`, and
a task `query_str`.
This includes both a high-level description of the table
as well as a description of each column in the table.

"""
RefineTableContextPrompt = PromptTemplate

"""Define the knowledge graph triplet extraction prompt."""
KnowledgeGraphPrompt = PromptTemplate

"""Simple Input prompt.

Required template variables: `query_str`.
"""
SimpleInputPrompt = PromptTemplate

"""Pandas prompt. Convert query to python code.

Required template variables: `query_str`, `df_str`, `instruction_str`.
"""
PandasPrompt = PromptTemplate


"""Choice select prompt. Select from a list of choices.

Required template variables: `context_str`, `query_str`.
"""
ChoiceSelectPrompt = PromptTemplate
