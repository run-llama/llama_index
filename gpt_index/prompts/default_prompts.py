"""Set of default prompts."""

from gpt_index.prompts.prompts import (
    KeywordExtractPrompt,
    QueryKeywordExtractPrompt,
    QuestionAnswerPrompt,
    RefinePrompt,
    SummaryPrompt,
    TreeInsertPrompt,
    TreeSelectMultiplePrompt,
    TreeSelectPrompt,
)

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

DEFAULT_SUMMARY_PROMPT = SummaryPrompt(DEFAULT_SUMMARY_PROMPT_TMPL)

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
DEFAULT_INSERT_PROMPT = TreeInsertPrompt(DEFAULT_INSERT_PROMPT_TMPL)


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
DEFAULT_QUERY_PROMPT = TreeSelectPrompt(DEFAULT_QUERY_PROMPT_TMPL)

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
DEFAULT_QUERY_PROMPT_MULTIPLE = TreeSelectMultiplePrompt(
    DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL
)


DEFAULT_REFINE_PROMPT_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer."
)
DEFAULT_REFINE_PROMPT = RefinePrompt(DEFAULT_REFINE_PROMPT_TMPL)


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)
DEFAULT_TEXT_QA_PROMPT = QuestionAnswerPrompt(DEFAULT_TEXT_QA_PROMPT_TMPL)


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
DEFAULT_KEYWORD_EXTRACT_TEMPLATE = KeywordExtractPrompt(
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL
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
DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE = QueryKeywordExtractPrompt(
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL
)
