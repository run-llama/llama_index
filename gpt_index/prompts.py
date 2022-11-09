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

# # single choice
DEFAULT_QUERY_PROMPT = (
    "Some choices are given below. It is provided in a numbered list (1 to {num_chunks}),"
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return the choice that "
    "is most relevant to the question: '{query_str}'\n"
    "Provide choice in the following format: 'ANSWER: <number>' and explain why "
    "this summary was selected in relation to the question.\n"
)

# multiple choice
# DEFAULT_QUERY_PROMPT = (
#     "Some choices are given below. It is provided in a numbered list (1 to {num_chunks}),"
#     "where each item in the list corresponds to a summary.\n"
#     "---------------------\n"
#     "{context_list}"
#     "\n---------------------\n"
#     "Using only the choices above and not prior knowledge, return the top choices "
#     "(no more than 3, ranked by most relevant to least) that "
#     "are most relevant to the question: '{query_str}'\n"
#     "Provide choices in the following format: 'ANSWER: <numbers>' and explain why "
#     "these summaries were selected in relation to the question.\n"
# )


DEFAULT_TEXT_QA_PROMPT = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, answer the question: {query_str}\n"
)