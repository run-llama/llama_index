"""Default query for GPTSQLStructStoreIndex."""
import logging
from typing import Any, Optional

from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.struct_store.container_builder import (
    SQLContextContainerBuilder,
)
from llama_index.indices.struct_store.sql import GPTSQLStructStoreIndex
from llama_index.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.prompts.prompts import TextToSQLPrompt
from llama_index.response.schema import Response
from llama_index.token_counter.token_counter import llm_token_counter

logger = logging.getLogger(__name__)


class GPTSQLStructStoreQueryEngine(BaseQueryEngine):
    """GPT SQL query engine over a structured database.

    Runs raw SQL over a GPTSQLStructStoreIndex. No LLM calls are made here.
    NOTE: this query cannot work with composed indices - if the index
    contains subindices, those subindices will not be queried.
    """

    def __init__(
        self,
        index: GPTSQLStructStoreIndex,
        sql_context_container: Optional[SQLContextContainerBuilder] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._sql_database = index.sql_database
        self._sql_context_container = (
            sql_context_container or index.sql_context_container
        )

    def _query(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        # NOTE: override query method in order to fetch the right results.
        # NOTE: since the query_str is a SQL query, it doesn't make sense
        # to use ResponseBuilder anywhere.
        response_str, extra_info = self._sql_database.run_sql(query_bundle.query_str)
        response = Response(response=response_str, extra_info=extra_info)
        return response

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        return self._query(query_bundle)


class GPTNLStructStoreQueryEngine(BaseQueryEngine):
    """GPT natural language query engine over a structured database.

    Given a natural language query, we will extract the query to SQL.
    Runs raw SQL over a GPTSQLStructStoreIndex. No LLM calls are made during
    the SQL execution.
    NOTE: this query cannot work with composed indices - if the index
    contains subindices, those subindices will not be queried.
    """

    def __init__(
        self,
        index: GPTSQLStructStoreIndex,
        text_to_sql_prompt: Optional[TextToSQLPrompt] = None,
        context_query_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._sql_database = index.sql_database
        self._sql_context_container = index.sql_context_container
        self._service_context = index.service_context
        self._ref_doc_id_column = index.ref_doc_id_column

        self._text_to_sql_prompt = text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
        self._context_query_kwargs = context_query_kwargs or {}

    def _parse_response_to_sql(self, response: str) -> str:
        """Parse response to SQL."""
        result_response = response.strip()
        return result_response

    def _get_table_context(self, query_bundle: QueryBundle) -> str:
        """Get table context.

        Get tables schema + optional context as a single string. Taken from
        SQLContextContainer.

        """
        if self._sql_context_container.context_str is not None:
            tables_desc_str = self._sql_context_container.context_str
        else:
            table_desc_list = []
            context_dict = self._sql_context_container.context_dict
            if context_dict is None:
                raise ValueError(
                    "context_dict must be provided. There is currently no "
                    "table context."
                )
            for table_desc in context_dict.values():
                table_desc_list.append(table_desc)
            tables_desc_str = "\n\n".join(table_desc_list)

        return tables_desc_str

    @llm_token_counter("query")
    def _query(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        table_desc_str = self._get_table_context(query_bundle)
        logger.info(f"> Table desc str: {table_desc_str}")

        response_str, _ = self._service_context.llm_predictor.predict(
            self._text_to_sql_prompt,
            query_str=query_bundle.query_str,
            schema=table_desc_str,
            dialect=self._sql_database.dialect,
        )

        sql_query_str = self._parse_response_to_sql(response_str)
        # assume that it's a valid SQL query
        logger.debug(f"> Predicted SQL query: {sql_query_str}")

        response_str, extra_info = self._sql_database.run_sql(sql_query_str)
        extra_info["sql_query"] = sql_query_str
        response = Response(response=response_str, extra_info=extra_info)
        return response

    @llm_token_counter("aquery")
    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        table_desc_str = self._get_table_context(query_bundle)
        logger.info(f"> Table desc str: {table_desc_str}")

        (
            response_str,
            formatted_prompt,
        ) = await self._service_context.llm_predictor.apredict(
            self._text_to_sql_prompt,
            query_str=query_bundle.query_str,
            schema=table_desc_str,
            dialect=self._sql_database.dialect,
        )

        sql_query_str = self._parse_response_to_sql(response_str)
        # assume that it's a valid SQL query
        logger.debug(f"> Predicted SQL query: {sql_query_str}")

        response_str, extra_info = self._sql_database.run_sql(sql_query_str)
        extra_info["sql_query"] = sql_query_str
        response = Response(response=response_str, extra_info=extra_info)
        return response
