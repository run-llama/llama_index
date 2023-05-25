"""Set of default prompts."""

from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType

############################################
# Tree
############################################

DEFAULT_SUMMARY_PROMPT_TMPL = (
    "Write a summary of the following. Try to use only the "
    "information provided. "
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'SUMMARY:"""\n'
)

DEFAULT_SUMMARY_PROMPT = Prompt(
    DEFAULT_SUMMARY_PROMPT_TMPL, prompt_type=PromptType.SUMMARY
)

# insert prompts
DEFAULT_INSERT_PROMPT_TMPL = (
    "Context information is below. It is provided in a numbered list "
    "(1 to {num_chunks}),"
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "---------------------\n"
    "Given the context information, here is a new piece of "
    "information: {new_chunk_text}\n"
    "Answer with the number corresponding to the summary that should be updated. "
    "The answer should be the number corresponding to the "
    "summary that is most relevant to the question.\n"
)
DEFAULT_INSERT_PROMPT = Prompt(
    DEFAULT_INSERT_PROMPT_TMPL, prompt_type=PromptType.TREE_INSERT
)


# # single choice
DEFAULT_QUERY_PROMPT_TMPL = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_chunks}),"
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return "
    "the choice that is most relevant to the question: '{query_str}'\n"
    "Provide choice in the following format: 'ANSWER: <number>' and explain why "
    "this summary was selected in relation to the question.\n"
)
DEFAULT_QUERY_PROMPT = Prompt(
    DEFAULT_QUERY_PROMPT_TMPL, prompt_type=PromptType.TREE_SELECT
)

# multiple choice
DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL = (
    "Some choices are given below. It is provided in a numbered "
    "list (1 to {num_chunks}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return the top choices "
    "(no more than {branching_factor}, ranked by most relevant to least) that "
    "are most relevant to the question: '{query_str}'\n"
    "Provide choices in the following format: 'ANSWER: <numbers>' and explain why "
    "these summaries were selected in relation to the question.\n"
)
DEFAULT_QUERY_PROMPT_MULTIPLE = Prompt(
    DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL, prompt_type=PromptType.TREE_SELECT_MULTIPLE
)


DEFAULT_REFINE_PROMPT_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer."
)
DEFAULT_REFINE_PROMPT = Prompt(
    DEFAULT_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)
DEFAULT_TEXT_QA_PROMPT = Prompt(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)


############################################
# Keyword Table
############################################

DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "Some text is provided below. Given the text, extract up to {max_keywords} "
    "keywords from the text. Avoid stopwords."
    "---------------------\n"
    "{text}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
)
DEFAULT_KEYWORD_EXTRACT_TEMPLATE = Prompt(
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL, prompt_type=PromptType.KEYWORD_EXTRACT
)


# NOTE: the keyword extraction for queries can be the same as
# the one used to build the index, but here we tune it to see if performance is better.
DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "A question is provided below. Given the question, extract up to {max_keywords} "
    "keywords from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
)
DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE = Prompt(
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)


############################################
# Structured Store
############################################

DEFAULT_SCHEMA_EXTRACT_TMPL = (
    "We wish to extract relevant fields from an unstructured text chunk into "
    "a structured schema. We first provide the unstructured text, and then "
    "we provide the schema that we wish to extract. "
    "-----------text-----------\n"
    "{text}\n"
    "-----------schema-----------\n"
    "{schema}\n"
    "---------------------\n"
    "Given the text and schema, extract the relevant fields from the text in "
    "the following format: "
    "field1: <value>\nfield2: <value>\n...\n\n"
    "If a field is not present in the text, don't include it in the output."
    "If no fields are present in the text, return a blank string.\n"
    "Fields: "
)
DEFAULT_SCHEMA_EXTRACT_PROMPT = Prompt(
    DEFAULT_SCHEMA_EXTRACT_TMPL, prompt_type=PromptType.SCHEMA_EXTRACT
)

# NOTE: taken from langchain and adapted
# https://tinyurl.com/b772sd77
DEFAULT_TEXT_TO_SQL_TMPL = (
    "Given an input question, first create a syntactically correct {dialect} "
    "query to run, then look at the results of the query and return the answer. "
    "You can order the results by a relevant column to return the most "
    "interesting examples in the database.\n"
    "Never query for all the columns from a specific table, only ask for a the "
    "few relevant columns given the question.\n"
    "Pay attention to use only the column names that you can see in the schema "
    "description. "
    "Be careful to not query for columns that do not exist. "
    "Pay attention to which column is in which table. "
    "Also, qualify column names with the table name when needed.\n"
    "Use the following format:\n"
    "Question: Question here\n"
    "SQLQuery: SQL Query to run\n"
    "SQLResult: Result of the SQLQuery\n"
    "Answer: Final answer here\n"
    "Only use the tables listed below.\n"
    "{schema}\n"
    "Question: {query_str}\n"
    "SQLQuery: "
)

DEFAULT_TEXT_TO_SQL_PROMPT = Prompt(
    DEFAULT_TEXT_TO_SQL_TMPL,
    stop_token="\nSQLResult:",
    prompt_type=PromptType.TEXT_TO_SQL,
)


# NOTE: by partially filling schema, we can reduce to a QuestionAnswer prompt
# that we can feed to ur table
DEFAULT_TABLE_CONTEXT_TMPL = (
    "We have provided a table schema below. "
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "We have also provided context information below. "
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and the table schema, "
    "give a response to the following task: {query_str}"
)

DEFAULT_TABLE_CONTEXT_QUERY = (
    "Provide a high-level description of the table, "
    "as well as a description of each column in the table. "
    "Provide answers in the following format:\n"
    "TableDescription: <description>\n"
    "Column1Description: <description>\n"
    "Column2Description: <description>\n"
    "...\n\n"
)

DEFAULT_TABLE_CONTEXT_PROMPT = Prompt(
    DEFAULT_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
)

# NOTE: by partially filling schema, we can reduce to a RefinePrompt
# that we can feed to ur table
DEFAULT_REFINE_TABLE_CONTEXT_TMPL = (
    "We have provided a table schema below. "
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "We have also provided some context information below. "
    "{context_msg}\n"
    "---------------------\n"
    "Given the context information and the table schema, "
    "give a response to the following task: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer."
)
DEFAULT_REFINE_TABLE_CONTEXT_PROMPT = Prompt(
    DEFAULT_REFINE_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
)


############################################
# Knowledge-Graph Table
############################################

DEFAULT_KG_TRIPLET_EXTRACT_TMPL = (
    "Some text is provided below. Given the text, extract up to "
    "{max_knowledge_triplets} "
    "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
    "---------------------\n"
    "Example:"
    "Text: Alice is Bob's mother."
    "Triplets:\n(Alice, is mother of, Bob)\n"
    "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    "Triplets:\n"
    "(Philz, is, coffee shop)\n"
    "(Philz, founded in, Berkeley)\n"
    "(Philz, founded in, 1982)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)
DEFAULT_KG_TRIPLET_EXTRACT_PROMPT = Prompt(
    DEFAULT_KG_TRIPLET_EXTRACT_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)

############################################
# HYDE
##############################################

HYDE_TMPL = (
    "Please write a passage to answer the question\n"
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'Passage:"""\n'
)

DEFAULT_HYDE_PROMPT = Prompt(HYDE_TMPL, prompt_type=PromptType.SUMMARY)


############################################
# Simple Input
############################################

DEFAULT_SIMPLE_INPUT_TMPL = "{query_str}"
DEFAULT_SIMPLE_INPUT_PROMPT = Prompt(
    DEFAULT_SIMPLE_INPUT_TMPL, prompt_type=PromptType.SIMPLE_INPUT
)


############################################
# Pandas
############################################

DEFAULT_PANDAS_TMPL = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Here is the input query: {query_str}.\n"
    "Given the df information and the input query, please follow "
    "these instructions:\n"
    "{instruction_str}"
    "Output:\n"
)

DEFAULT_PANDAS_PROMPT = Prompt(DEFAULT_PANDAS_TMPL, prompt_type=PromptType.PANDAS)
