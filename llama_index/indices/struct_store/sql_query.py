"""Default query for SQLStructStoreIndex."""
import logging
from typing import Any, List, Optional, Union

from sqlalchemy import Table

from abc import abstractmethod
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.struct_store.container_builder import (
    SQLContextContainerBuilder,
)
from llama_index.indices.struct_store.sql import SQLStructStoreIndex
from llama_index.langchain_helpers.sql_wrapper import SQLDatabase
from llama_index.prompts.base import Prompt
from llama_index.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.prompts.prompt_type import PromptType
from llama_index.response.schema import Response
from llama_index.token_counter.token_counter import llm_token_counter
from llama_index.objects.table_node_mapping import SQLTableSchema
from llama_index.objects.base import ObjectRetriever

logger = logging.getLogger(__name__)


DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {sql_response_str}\n"
    "Response: "
)
DEFAULT_RESPONSE_SYNTHESIS_PROMPT = Prompt(
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL,
    prompt_type=PromptType.SQL_RESPONSE_SYNTHESIS,
)


class SQLStructStoreQueryEngine(BaseQueryEngine):
    """GPT SQL query engine over a structured database.

    NOTE: deprecated, kept for backward compatibility

    Runs raw SQL over a SQLStructStoreIndex. No LLM calls are made here.
    NOTE: this query cannot work with composed indices - if the index
    contains subindices, those subindices will not be queried.
    """

    def __init__(
        self,
        index: SQLStructStoreIndex,
        sql_context_container: Optional[SQLContextContainerBuilder] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._sql_database = index.sql_database
        self._sql_context_container = (
            sql_context_container or index.sql_context_container
        )
        super().__init__(index.service_context.callback_manager)

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


class NLStructStoreQueryEngine(BaseQueryEngine):
    """GPT natural language query engine over a structured database.

    NOTE: deprecated, kept for backward compatibility

    Given a natural language query, we will extract the query to SQL.
    Runs raw SQL over a SQLStructStoreIndex. No LLM calls are made during
    the SQL execution.
    NOTE: this query cannot work with composed indices - if the index
    contains subindices, those subindices will not be queried.

    Args:
        index (SQLStructStoreIndex): A SQL Struct Store Index
        text_to_sql_prompt (Optional[Prompt]): A Text to SQL Prompt
            to use for the query. Defaults to DEFAULT_TEXT_TO_SQL_PROMPT.
        context_query_kwargs (Optional[dict]): Keyword arguments for the
            context query. Defaults to {}.
        synthesize_response (bool): Whether to synthesize a response from the
            query results. Defaults to True.
        response_synthesis_prompt (Optional[Prompt]): A
            Response Synthesis Prompt to use for the query. Defaults to
            DEFAULT_RESPONSE_SYNTHESIS_PROMPT.
    """

    def __init__(
        self,
        index: SQLStructStoreIndex,
        text_to_sql_prompt: Optional[Prompt] = None,
        context_query_kwargs: Optional[dict] = None,
        synthesize_response: bool = True,
        response_synthesis_prompt: Optional[Prompt] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._sql_database = index.sql_database
        self._sql_context_container = index.sql_context_container
        self._service_context = index.service_context
        self._ref_doc_id_column = index.ref_doc_id_column

        self._text_to_sql_prompt = text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
        self._response_synthesis_prompt = (
            response_synthesis_prompt or DEFAULT_RESPONSE_SYNTHESIS_PROMPT
        )
        self._context_query_kwargs = context_query_kwargs or {}
        self._synthesize_response = synthesize_response
        super().__init__(index.service_context.callback_manager)

    @property
    def service_context(self) -> ServiceContext:
        """Get service context."""
        return self._service_context

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

        raw_response_str, extra_info = self._sql_database.run_sql(sql_query_str)
        extra_info["sql_query"] = sql_query_str

        if self._synthesize_response:
            response_str, _ = self._service_context.llm_predictor.predict(
                self._response_synthesis_prompt,
                query_str=query_bundle.query_str,
                sql_query=sql_query_str,
                sql_response_str=raw_response_str,
            )
        else:
            response_str = raw_response_str

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


class BaseSQLTableQueryEngine(BaseQueryEngine):
    def __init__(
        self,
        sql_database: SQLDatabase,
        text_to_sql_prompt: Optional[Prompt] = None,
        context_query_kwargs: Optional[dict] = None,
        synthesize_response: bool = True,
        response_synthesis_prompt: Optional[Prompt] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._sql_database = sql_database
        self._service_context = service_context or ServiceContext.from_defaults()

        self._text_to_sql_prompt = text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
        self._response_synthesis_prompt = (
            response_synthesis_prompt or DEFAULT_RESPONSE_SYNTHESIS_PROMPT
        )
        self._context_query_kwargs = context_query_kwargs or {}
        self._synthesize_response = synthesize_response
        super().__init__(self._service_context.callback_manager, **kwargs)

    @property
    def service_context(self) -> ServiceContext:
        """Get service context."""
        return self._service_context

    def _parse_response_to_sql(self, response: str) -> str:
        """Parse response to SQL."""
        result_response = response.strip()
        return result_response

    @abstractmethod
    def _get_table_context(self, query_bundle: QueryBundle) -> str:
        """Get table context.

        Get tables schema + optional context as a single string.

        """

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

        raw_response_str, extra_info = self._sql_database.run_sql(sql_query_str)
        extra_info["sql_query"] = sql_query_str

        if self._synthesize_response:
            response_str, _ = self._service_context.llm_predictor.predict(
                self._response_synthesis_prompt,
                query_str=query_bundle.query_str,
                sql_query=sql_query_str,
                sql_response_str=raw_response_str,
            )
        else:
            response_str = raw_response_str

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


class NLSQLTableQueryEngine(BaseSQLTableQueryEngine):
    """NL SQL Table query engine."""

    def __init__(
        self,
        sql_database: SQLDatabase,
        text_to_sql_prompt: Optional[Prompt] = None,
        context_query_kwargs: Optional[dict] = None,
        synthesize_response: bool = True,
        response_synthesis_prompt: Optional[Prompt] = None,
        tables: Optional[Union[List[str], List[Table]]] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._tables = tables
        super().__init__(
            sql_database,
            text_to_sql_prompt=text_to_sql_prompt,
            context_query_kwargs=context_query_kwargs,
            synthesize_response=synthesize_response,
            response_synthesis_prompt=response_synthesis_prompt,
            service_context=service_context,
            **kwargs,
        )

    def _get_table_context(self, query_bundle: QueryBundle) -> str:
        """Get table context.

        Get tables schema + optional context as a single string.

        """
        context_strs = []
        if self._tables:
            for table in self._tables:
                if isinstance(table, Table):
                    table_str = str(table.name)
                if isinstance(table, str):
                    table_str = table
                else:
                    raise ValueError(f"Unknown table type: {table}")
                table_info = self._sql_database.get_single_table_info(table_str)
                context_strs.append(table_info)

        else:
            # get all tables
            table_names = self._sql_database.get_table_names()
            for table_name in table_names:
                table_info = self._sql_database.get_single_table_info(table_name)
                context_strs.append(table_info)

        tables_desc_str = "\n\n".join(context_strs)
        return tables_desc_str


class SQLTableRetrieverQueryEngine(BaseSQLTableQueryEngine):
    """SQL Table retriever query engine."""

    def __init__(
        self,
        sql_database: SQLDatabase,
        table_retriever: ObjectRetriever[SQLTableSchema],
        text_to_sql_prompt: Optional[Prompt] = None,
        context_query_kwargs: Optional[dict] = None,
        synthesize_response: bool = True,
        response_synthesis_prompt: Optional[Prompt] = None,
        service_context: Optional[ServiceContext] = None,
        context_str_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._table_retriever = table_retriever
        self._context_str_prefix = context_str_prefix
        super().__init__(
            sql_database,
            text_to_sql_prompt=text_to_sql_prompt,
            context_query_kwargs=context_query_kwargs,
            synthesize_response=synthesize_response,
            response_synthesis_prompt=response_synthesis_prompt,
            service_context=service_context,
            **kwargs,
        )

    def _get_table_context(self, query_bundle: QueryBundle) -> str:
        """Get table context.

        Get tables schema + optional context as a single string.

        """
        if self._context_str_prefix is not None:
            context_strs = [self._context_str_prefix]

        # TODO: allow top-level context
        table_schema_objs = self._table_retriever.retrieve(query_bundle)
        context_strs = []
        for table_schema_obj in table_schema_objs:
            table_info = self._sql_database.get_single_table_info(
                table_schema_obj.table_name
            )
            context_strs.append(table_info)

        tables_desc_str = "\n\n".join(context_strs)
        return tables_desc_str


# legacy
GPTNLStructStoreQueryEngine = NLStructStoreQueryEngine
GPTSQLStructStoreQueryEngine = SQLStructStoreQueryEngine
