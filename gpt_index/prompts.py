"""Set of default prompts."""

DEFAULT_SUMMARY_PROMPT = (
    "Write a concise summary of the following:\n"
    "\n"
    "\n"
    "{text}\n"
    "\n"
    "\n"
    "CONCISE SUMMARY:\"\"\"\n"
)

DEFAULT_INSERT_PROMPT = (
    "Context information is below. It is provided in a numbered list (1 to {num_chunks}),"
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "---------------------\n"
    "Given the context information, here is a new piece of information: {new_chunk_text}\n"
    "Answer with the number corresponding to the summary that should be updated. "
    "The answer should be the number corresponding to the "
    "summary that is most relevant to the question.\n"
)

DEFAULT_QUERY_PROMPT = (
    "Context information is below. It is provided in a numbered list (1 to {num_chunks}),"
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "---------------------\n"
    "Given the context information, answer the question: {query_str}\n"
    "The answer should be the number corresponding to the "
    "summary that is most relevant to the question.\n"
)

DEFAULT_TEXT_QA_PROMPT = (
    "Context information is below. "
    "---------------------\n"
    "{context_str}"
    "---------------------\n"
    "Given the context information, answer the question: {query_str}\n"
)