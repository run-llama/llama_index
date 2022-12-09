"""Mock prompt utils."""

from gpt_index.prompts.base import Prompt

MOCK_SUMMARY_PROMPT_TMPL = "{text}\n"
MOCK_SUMMARY_PROMPT = Prompt(
    input_variables=["text"], template=MOCK_SUMMARY_PROMPT_TMPL
)

MOCK_INSERT_PROMPT_TMPL = "{num_chunks}\n{context_list}{new_chunk_text}\n"
MOCK_INSERT_PROMPT = Prompt(
    input_variables=["num_chunks", "context_list", "new_chunk_text"],
    template=MOCK_INSERT_PROMPT_TMPL,
)

# # single choice
MOCK_QUERY_PROMPT_TMPL = "{num_chunks}\n" "{context_list}\n" "{query_str}'\n"
MOCK_QUERY_PROMPT = Prompt(
    input_variables=["num_chunks", "context_list", "query_str"],
    template=MOCK_QUERY_PROMPT_TMPL,
)


MOCK_REFINE_PROMPT_TMPL = "{query_str}\n" "{existing_answer}\n" "{context_msg}\n"
MOCK_REFINE_PROMPT = Prompt(
    input_variables=["query_str", "existing_answer", "context_msg"],
    template=MOCK_REFINE_PROMPT_TMPL,
)


MOCK_TEXT_QA_PROMPT_TMPL = "{context_str}\n" "{query_str}\n"
MOCK_TEXT_QA_PROMPT = Prompt(
    input_variables=["context_str", "query_str"], template=MOCK_TEXT_QA_PROMPT_TMPL
)


MOCK_KEYWORD_EXTRACT_PROMPT_TMPL = "{max_keywords}\n{text}\n"
MOCK_KEYWORD_EXTRACT_PROMPT = Prompt(
    input_variables=["max_keywords", "text"], template=MOCK_KEYWORD_EXTRACT_PROMPT_TMPL
)

# TODO: consolidate with keyword extract
MOCK_QUERY_KEYWORD_EXTRACT_PROMPT_TMPL = "{max_keywords}\n{question}\n"
MOCK_QUERY_KEYWORD_EXTRACT_PROMPT = Prompt(
    input_variables=["max_keywords", "question"],
    template=MOCK_QUERY_KEYWORD_EXTRACT_PROMPT_TMPL,
)
