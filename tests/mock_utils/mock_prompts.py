"""Mock prompt utils."""

from gpt_index.prompts.prompts import (
    KeywordExtractPrompt,
    KnowledgeGraphPrompt,
    QueryKeywordExtractPrompt,
    QuestionAnswerPrompt,
    RefinePrompt,
    SchemaExtractPrompt,
    SummaryPrompt,
    TableContextPrompt,
    TextToSQLPrompt,
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


MOCK_SCHEMA_EXTRACT_PROMPT_TMPL = "{text}\n{schema}"
MOCK_SCHEMA_EXTRACT_PROMPT = SchemaExtractPrompt(MOCK_SCHEMA_EXTRACT_PROMPT_TMPL)

MOCK_TEXT_TO_SQL_PROMPT_TMPL = "{schema}\n{query_str}"
MOCK_TEXT_TO_SQL_PROMPT = TextToSQLPrompt(MOCK_TEXT_TO_SQL_PROMPT_TMPL)


MOCK_TABLE_CONTEXT_PROMPT_TMPL = "{schema}\n{context_str}\n{query_str}"
MOCK_TABLE_CONTEXT_PROMPT = TableContextPrompt(MOCK_TABLE_CONTEXT_PROMPT_TMPL)

MOCK_KG_TRIPLET_EXTRACT_PROMPT_TMPL = "{max_knowledge_triplets}\n{text}"
MOCK_KG_TRIPLET_EXTRACT_PROMPT = KnowledgeGraphPrompt(
    MOCK_KG_TRIPLET_EXTRACT_PROMPT_TMPL
)
