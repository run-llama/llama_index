"""Default query for GPTFaissIndex."""
import logging
from typing import Any, Optional, cast

from gpt_index.data_structs.table import SQLStructTable
from gpt_index.indices.common.struct_store.schema import SQLContextContainer
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.query_runner import QueryRunner
from gpt_index.indices.query.schema import QueryBundle, QueryConfig, QueryMode
from gpt_index.indices.registry import IndexRegistry
from gpt_index.langchain_helpers.sql_wrapper import SQLDatabase
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from gpt_index.prompts.prompts import TextToSQLPrompt
from gpt_index.response.schema import Response
from gpt_index.token_counter.token_counter import llm_token_counter

DEFAULT_CONTEXT_QUERY_TMPL = (
    "Please return the relevant tables for the following query: {orig_query_str}"
)


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
        # TODO: remove table_context_str and consolidate with sql_context_container
        table_context_str: Optional[str] = None,
        ref_doc_id_column: Optional[str] = None,
        text_to_sql_prompt: Optional[TextToSQLPrompt] = None,
        context_query_tmpl: str = DEFAULT_CONTEXT_QUERY_TMPL,
        context_query_mode: Optional[str] = None,
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
        self._table_context_str = table_context_str
        self._ref_doc_id_column = ref_doc_id_column
        self._text_to_sql_prompt = text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
        # self._context_query_config = context_query_config
        self._context_query_mode = context_query_mode
        self._context_query_kwargs = context_query_kwargs or {}
        self._context_query_tmpl = context_query_tmpl

    def _parse_response_to_sql(self, response: str) -> str:
        """Parse response to SQL."""
        result_response = response.strip()
        return result_response

    # def _get_all_tables_desc(self) -> str:
    #     """Get tables schema + optional context as a single string."""
    #     tables_desc = []
    #     for table_name in self._sql_database.get_table_names():
    #         table_desc = self._sql_database.get_single_table_info(table_name)
    #         table_text = f"Schema of table {table_name}:\n" f"{table_desc}\n"
    #         if table_name in self._index_struct.context_dict:
    #             table_text += f"Context of table {table_name}:\n"
    #             table_text += self._index_struct.context_dict[table_name]
    #         tables_desc.append(table_text)
    #     return "\n\n".join(tables_desc)

    def _get_table_context(self, query_bundle: QueryBundle) -> str:
        """Get table context.

        Get tables schema + optional context as a single string. Taken from
        SQLContextContainer.

        If _table_context_str is not None, return that. Otherwise,
        return the relevant context from the SQLContextContainer.

        """
        if self._table_context_str is not None:
            return self._table_context_str

        if self._sql_context_container.index_struct is None:
            table_desc_list = []
            for table_desc in self._sql_context_container.context_dict.values():
                table_desc_list.append(table_desc)
            tables_desc_str = "\n\n".join(table_desc_list)
        else:
            # TODO: make query_str better
            orig_query_str = query_bundle.query_str
            context_query_str = self._context_query_tmpl.format(
                orig_query_str=orig_query_str
            )
            # TODO/NOTE: instead of using the existing queryrunner on the query class,
            # construct a new one
            mode_enum = QueryMode(self._context_query_mode)
            query_config = QueryConfig(
                index_struct_type=self._index_struct.get_type(),
                query_mode=mode_enum,
                query_kwargs=self._context_query_kwargs,
            )
            query_runner = QueryRunner(
                self._llm_predictor,
                self._prompt_helper,
                self._embed_model,
                self._docstore,
                # NOTE: this is hacky, it's a way of passing current index registry
                cast(IndexRegistry, getattr(self._query_runner, "index_registry")),
                query_configs=[query_config],
                query_transform=None,
                recursive=False,
                use_async=self._use_async,
            )
            response = query_runner.query(context_query_str, self._index_struct)
            tables_desc_str = response.response

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
        response = Response(response=response_str, extra_info=extra_info)
        return response
