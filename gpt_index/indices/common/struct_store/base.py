"""Common classes for structured operations."""

from typing import Dict, List, Optional, Sequence

from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.response.builder import ResponseBuilder, TextChunk
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.sql_wrapper import SQLDatabase
from gpt_index.prompts.default_prompts import (
    DEFAULT_REFINE_TABLE_CONTEXT_PROMPT,
    DEFAULT_TABLE_CONTEXT_PROMPT,
    DEFAULT_TABLE_CONTEXT_QUERY,
)
from gpt_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    RefineTableContextPrompt,
    TableContextPrompt,
)
from gpt_index.schema import BaseDocument


class SQLContextBuilder:
    """Builder that builds context for a given set of SQL tables.

    Args:
        sql_database (Optional[SQLDatabase]): SQL database to use,
        context_builder_prompt (Optional[TableContextPrompt]): A
            Table Context Prompt (see :ref:`Prompt-Templates`).
    """

    def __init__(
        self,
        sql_database: SQLDatabase,
        llm_predictor: Optional[LLMPredictor] = None,
        prompt_helper: Optional[PromptHelper] = None,
        table_context_prompt: Optional[TableContextPrompt] = None,
        refine_table_context_prompt: Optional[RefineTableContextPrompt] = None,
        table_context_task: Optional[str] = None,
    ) -> None:
        """Initialize params."""
        # TODO: take in an entire index instead of forming a response builder
        if sql_database is None:
            raise ValueError("sql_database must be provided.")
        self._sql_database = sql_database
        self._llm_predictor = llm_predictor or LLMPredictor()
        self._prompt_helper = prompt_helper or PromptHelper.from_llm_predictor(
            self._llm_predictor
        )
        self._table_context_prompt = (
            table_context_prompt or DEFAULT_TABLE_CONTEXT_PROMPT
        )
        self._refine_table_context_prompt = (
            refine_table_context_prompt or DEFAULT_REFINE_TABLE_CONTEXT_PROMPT
        )
        self._table_context_task = table_context_task or DEFAULT_TABLE_CONTEXT_QUERY

    def build_all_context_from_documents(
        self,
        documents_dict: Dict[str, List[BaseDocument]],
    ) -> Dict[str, str]:
        """Build context for all tables in the database."""
        context_dict = {}
        for table_name in self._sql_database.get_table_names():
            context_dict[table_name] = self.build_table_context_from_documents(
                documents_dict[table_name], table_name
            )
        return context_dict

    def build_table_context_from_documents(
        self,
        documents: Sequence[BaseDocument],
        table_name: str,
    ) -> str:
        """Build context from documents for a single table."""
        schema = self._sql_database.get_single_table_info(table_name)
        prompt_with_schema = QuestionAnswerPrompt.from_prompt(
            self._table_context_prompt.partial_format(schema=schema)
        )
        refine_prompt_with_schema = RefinePrompt.from_prompt(
            self._refine_table_context_prompt.partial_format(schema=schema)
        )
        text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
            prompt_with_schema, 1
        )
        # we use the ResponseBuilder to iteratively go through all texts
        response_builder = ResponseBuilder(
            self._prompt_helper,
            self._llm_predictor,
            prompt_with_schema,
            refine_prompt_with_schema,
        )
        for doc in documents:
            text_chunks = text_splitter.split_text(doc.get_text())
            for text_chunk in text_chunks:
                response_builder.add_text_chunks([TextChunk(text_chunk)])

        # feed in the "query_str" or the task
        table_context = response_builder.get_response(self._table_context_task)
        return table_context
