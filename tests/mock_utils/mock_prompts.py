"""Mock prompt utils."""

from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType

MOCK_SUMMARY_PROMPT_TMPL = "{context_str}\n"
MOCK_SUMMARY_PROMPT = Prompt(MOCK_SUMMARY_PROMPT_TMPL, prompt_type=PromptType.SUMMARY)

MOCK_INSERT_PROMPT_TMPL = "{num_chunks}\n{context_list}{new_chunk_text}\n"
MOCK_INSERT_PROMPT = Prompt(MOCK_INSERT_PROMPT_TMPL, prompt_type=PromptType.TREE_INSERT)

# # single choice
MOCK_QUERY_PROMPT_TMPL = "{num_chunks}\n" "{context_list}\n" "{query_str}'\n"
MOCK_QUERY_PROMPT = Prompt(MOCK_QUERY_PROMPT_TMPL, prompt_type=PromptType.TREE_SELECT)


MOCK_REFINE_PROMPT_TMPL = "{query_str}\n" "{existing_answer}\n" "{context_msg}\n"
MOCK_REFINE_PROMPT = Prompt(MOCK_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE)


MOCK_TEXT_QA_PROMPT_TMPL = "{context_str}\n" "{query_str}\n"
MOCK_TEXT_QA_PROMPT = Prompt(
    MOCK_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)


MOCK_KEYWORD_EXTRACT_PROMPT_TMPL = "{max_keywords}\n{text}\n"
MOCK_KEYWORD_EXTRACT_PROMPT = Prompt(
    MOCK_KEYWORD_EXTRACT_PROMPT_TMPL, prompt_type=PromptType.KEYWORD_EXTRACT
)

# TODO: consolidate with keyword extract
MOCK_QUERY_KEYWORD_EXTRACT_PROMPT_TMPL = "{max_keywords}\n{question}\n"
MOCK_QUERY_KEYWORD_EXTRACT_PROMPT = Prompt(
    MOCK_QUERY_KEYWORD_EXTRACT_PROMPT_TMPL, prompt_type=PromptType.QUERY_KEYWORD_EXTRACT
)


MOCK_SCHEMA_EXTRACT_PROMPT_TMPL = "{text}\n{schema}"
MOCK_SCHEMA_EXTRACT_PROMPT = Prompt(
    MOCK_SCHEMA_EXTRACT_PROMPT_TMPL, prompt_type=PromptType.SCHEMA_EXTRACT
)

MOCK_TEXT_TO_SQL_PROMPT_TMPL = "{dialect}\n{schema}\n{query_str}"
MOCK_TEXT_TO_SQL_PROMPT = Prompt(
    MOCK_TEXT_TO_SQL_PROMPT_TMPL, prompt_type=PromptType.TEXT_TO_SQL
)


MOCK_TABLE_CONTEXT_PROMPT_TMPL = "{schema}\n{context_str}\n{query_str}"
MOCK_TABLE_CONTEXT_PROMPT = Prompt(
    MOCK_TABLE_CONTEXT_PROMPT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
)

MOCK_KG_TRIPLET_EXTRACT_PROMPT_TMPL = "{max_knowledge_triplets}\n{text}"
MOCK_KG_TRIPLET_EXTRACT_PROMPT = Prompt(
    MOCK_KG_TRIPLET_EXTRACT_PROMPT_TMPL,
    prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT,
)

MOCK_INPUT_PROMPT_TMPL = "{query_str}"
MOCK_INPUT_PROMPT = Prompt(MOCK_INPUT_PROMPT_TMPL, prompt_type=PromptType.SIMPLE_INPUT)

MOCK_PANDAS_PROMPT_TMPL = "{query_str}\n{df_str}\n{instruction_str}"
MOCK_PANDAS_PROMPT = Prompt(MOCK_PANDAS_PROMPT_TMPL, prompt_type=PromptType.PANDAS)
