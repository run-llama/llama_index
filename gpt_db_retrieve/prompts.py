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

DEFAULT_QUERY_PROMPT = (
    "Context information is below. It is provided in a numbered list (1 to {num_chunks}),"
    "where each item in the list corresponds to a text chunk.\n"
    "---------------------\n"
    "{context_list}"
    "---------------------\n"
    "Given the context information, answer the question: {query_str}\n"
    "Given the question, choose the number (1 to {num_chunks}) corresponding to the context chunk "
    "that is most relevant to the question.\n"
    "Provide answer in the following format: \"ANSWER: <number>\"."
)