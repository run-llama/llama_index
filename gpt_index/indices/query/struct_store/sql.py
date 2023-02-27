"""Default query for GPTFaissIndex."""
import logging
from typing import Any, Optional

from gpt_index.data_structs.table import SQLStructTable
from gpt_index.indices.common.struct_store.schema import SQLContextContainer
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.schema import QueryBundle, QueryMode
from gpt_index.langchain_helpers.sql_wrapper import SQLDatabase
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from gpt_index.prompts.prompts import TextToSQLPrompt
from gpt_index.response.schema import Response
from gpt_index.token_counter.token_counter import llm_token_counter


class GPTSQLStructStoreIndexQuery(BaseGPTIndexQuery[SQLStructTable]):
    """GPT SQL query over a structured database.

    Runs raw SQL over a GPTSQLStructStoreIndex. No LLM calls are made here.
    NOTE: this query cannot work with composed indices - if the index
    contains subindices, those subindices will not be queried.

    .. code-block:: python

        response = index.query("<query_str>", mode="sql")

    """

    def __init__(
        self,
        index_struct: SQLStructTable,
        sql_database: Optional[SQLDatabase] = None,
        sql_context_container: Optional[SQLContextContainer] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct=index_struct, **kwargs)
        if sql_database is None:
            raise ValueError("sql_database must be provided.")
        self._sql_database = sql_database

    @llm_token_counter("query")
    def query(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        # NOTE: override query method in order to fetch the right results.
        # NOTE: since the query_str is a SQL query, it doesn't make sense
        # to use ResponseBuilder anywhere.
        response_str, extra_info = self._sql_database.run_sql(query_bundle.query_str)
        response = Response(response=response_str, extra_info=extra_info)
        return response


class GPTNLStructStoreIndexQuery(BaseGPTIndexQuery[SQLStructTable]):
    """GPT natural language query over a structured database.

    Given a natural language query, we will extract the query to SQL.
    Runs raw SQL over a GPTSQLStructStoreIndex. No LLM calls are made here.
    NOTE: this query cannot work with composed indices - if the index
    contains subindices, those subindices will not be queried.

    .. code-block:: python

        response = index.query("<query_str>", mode="sql")

    """

    def __init__(
        self,
        index_struct: SQLStructTable,
        sql_database: Optional[SQLDatabase] = None,
        sql_context_container: Optional[SQLContextContainer] = None,
        ref_doc_id_column: Optional[str] = None,
        text_to_sql_prompt: Optional[TextToSQLPrompt] = None,
        context_query_mode: QueryMode = QueryMode.DEFAULT,
        context_query_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct=index_struct, **kwargs)
        if sql_database is None:
            raise ValueError("sql_database must be provided.")
        self._sql_database = sql_database
        if sql_context_container is None:
            raise ValueError("sql_context_container must be provided.")
        self._sql_context_container = sql_context_container
        self._ref_doc_id_column = ref_doc_id_column
        self._text_to_sql_prompt = text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
        self._context_query_mode = context_query_mode
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

    def _query(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        table_desc_str = self._get_table_context(query_bundle)
        logging.info(f"> Table desc str: {table_desc_str}")
        response_str, _ = self._llm_predictor.predict(
            self._text_to_sql_prompt,
            query_str=query_bundle.query_str,
            schema=table_desc_str,
        )

        sql_query_str = self._parse_response_to_sql(response_str)
        # assume that it's a valid SQL query
        logging.debug(f"> Predicted SQL query: {sql_query_str}")

        response_str, extra_info = self._sql_database.run_sql(sql_query_str)
        extra_info["sql_query"] = sql_query_str
        response = Response(response=response_str, extra_info=extra_info)
        return response
