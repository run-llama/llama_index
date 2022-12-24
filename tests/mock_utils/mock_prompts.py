"""Mock prompt utils."""

from gpt_index.prompts.prompts import (
    KeywordExtractPrompt,
    QueryKeywordExtractPrompt,
    QuestionAnswerPrompt,
    RefinePrompt,
    SummaryPrompt,
    TreeInsertPrompt,
    TreeSelectPrompt,
)

MOCK_SUMMARY_PROMPT_TMPL = "{context_str}\n"
MOCK_SUMMARY_PROMPT = SummaryPrompt(MOCK_SUMMARY_PROMPT_TMPL)

MOCK_INSERT_PROMPT_TMPL = "{num_chunks}\n{context_list}{new_chunk_text}\n"
MOCK_INSERT_PROMPT = TreeInsertPrompt(MOCK_INSERT_PROMPT_TMPL)

# # single choice
MOCK_QUERY_PROMPT_TMPL = "{num_chunks}\n" "{context_list}\n" "{query_str}'\n"
MOCK_QUERY_PROMPT = TreeSelectPrompt(MOCK_QUERY_PROMPT_TMPL)


MOCK_REFINE_PROMPT_TMPL = "{query_str}\n" "{existing_answer}\n" "{context_msg}\n"
MOCK_REFINE_PROMPT = RefinePrompt(MOCK_REFINE_PROMPT_TMPL)


MOCK_TEXT_QA_PROMPT_TMPL = "{context_str}\n" "{query_str}\n"
MOCK_TEXT_QA_PROMPT = QuestionAnswerPrompt(MOCK_TEXT_QA_PROMPT_TMPL)


MOCK_KEYWORD_EXTRACT_PROMPT_TMPL = "{max_keywords}\n{text}\n"
MOCK_KEYWORD_EXTRACT_PROMPT = KeywordExtractPrompt(MOCK_KEYWORD_EXTRACT_PROMPT_TMPL)

# TODO: consolidate with keyword extract
MOCK_QUERY_KEYWORD_EXTRACT_PROMPT_TMPL = "{max_keywords}\n{question}\n"
MOCK_QUERY_KEYWORD_EXTRACT_PROMPT = QueryKeywordExtractPrompt(
    MOCK_QUERY_KEYWORD_EXTRACT_PROMPT_TMPL
)
