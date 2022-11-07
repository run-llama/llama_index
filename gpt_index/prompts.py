"""Set of default prompts."""

DEFAULT_SUMMARY_PROMPT = (
    "Write a summary of the following. Try to use only the information provided. "
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{text}\n"
    "\n"
    "\n"
    "SUMMARY:\"\"\"\n"
)

DEFAULT_QUERY_PROMPT = (
    "Some choices are given below. It is provided in a numbered list (1 to {num_chunks}),"
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "---------------------\n"
    "Return the choice that is most relevant to the question: {query_str}\n"
    # "Given the context information, answer the question: {query_str}\n"
    # "The answer should be the number corresponding to the "
    # "summary that is most relevant to the question. "
    "Provide answer in the following format: 'ANSWER: <number>' and explain why.\n"
)

DEFAULT_TEXT_QA_PROMPT = (
    "Context information is below. "
    "---------------------\n"
    "{context_str}"
    "---------------------\n"
    "Given the context information, answer the question: {query_str}\n"
)