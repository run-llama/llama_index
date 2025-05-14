"""SQL Retriever."""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.llms.llm import LLM
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.objects.table_node_mapping import SQLTableSchema
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_TEXT_TO_SQL_PROMPT,
)
from llama_index.core.prompts.mixin import (
    PromptDictType,
    PromptMixin,
    PromptMixinType,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType, TextNode
from llama_index.core.settings import Settings
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from sqlalchemy import Table

logger = logging.getLogger(__name__)


class SQLRetriever(BaseRetriever):
    """
    SQL Retriever.

    Retrieves via raw SQL statements.

    Args:
        sql_database (SQLDatabase): SQL database.
        return_raw (bool): Whether to return raw results or format results.
            Defaults to True.

    """

    def __init__(
        self,
        sql_database: SQLDatabase,
        return_raw: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._sql_database = sql_database
        self._return_raw = return_raw
        super().__init__(callback_manager)

    def _format_node_results(
        self, results: List[List[Any]], col_keys: List[str]
    ) -> List[NodeWithScore]:
        """Format node results."""
        nodes = []
        for result in results:
            # associate column keys with result tuple
            metadata = dict(zip(col_keys, result))
            # NOTE: leave text field blank for now
            text_node = TextNode(
                text="",
                metadata=metadata,
            )
            nodes.append(NodeWithScore(node=text_node))
        return nodes

    def retrieve_with_metadata(
        self, str_or_query_bundle: QueryType
    ) -> Tuple[List[NodeWithScore], Dict]:
        """Retrieve with metadata."""
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        raw_response_str, metadata = self._sql_database.run_sql(query_bundle.query_str)
        if self._return_raw:
            return [
                NodeWithScore(
                    node=TextNode(
                        text=raw_response_str,
                        metadata={
                            "sql_query": query_bundle.query_str,
                            "result": metadata["result"],
                            "col_keys": metadata["col_keys"],
                        },
                        excluded_embed_metadata_keys=[
                            "sql_query",
                            "result",
                            "col_keys",
                        ],
                        excluded_llm_metadata_keys=["sql_query", "result", "col_keys"],
                    )
                )
            ], metadata
        else:
            # return formatted
            results = metadata["result"]
            col_keys = metadata["col_keys"]
            return self._format_node_results(results, col_keys), metadata

    async def aretrieve_with_metadata(
        self, str_or_query_bundle: QueryType
    ) -> Tuple[List[NodeWithScore], Dict]:
        return self.retrieve_with_metadata(str_or_query_bundle)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        retrieved_nodes, _ = self.retrieve_with_metadata(query_bundle)
        return retrieved_nodes


class SQLParserMode(str, Enum):
    """SQL Parser Mode."""

    DEFAULT = "default"
    PGVECTOR = "pgvector"


class BaseSQLParser(DispatcherSpanMixin, ABC):
    """Base SQL Parser."""

    @abstractmethod
    def parse_response_to_sql(self, response: str, query_bundle: QueryBundle) -> str:
        """Parse response to SQL."""


class DefaultSQLParser(BaseSQLParser):
    """Default SQL Parser."""

    def parse_response_to_sql(self, response: str, query_bundle: QueryBundle) -> str:
        """Parse response to SQL."""
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            # TODO: move to removeprefix after Python 3.9+
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:") :]
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]
        return response.replace("```sql", "").replace("```", "").strip()


class PGVectorSQLParser(BaseSQLParser):
    """PGVector SQL Parser."""

    def __init__(
        self,
        embed_model: BaseEmbedding,
    ) -> None:
        """Initialize params."""
        self._embed_model = embed_model

    def parse_response_to_sql(self, response: str, query_bundle: QueryBundle) -> str:
        """Parse response to SQL."""
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            # TODO: move to removeprefix after Python 3.9+
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:") :]
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]

        # this gets you the sql string with [query_vector] placeholders
        raw_sql_str = response.strip().strip("```sql").strip("```").strip()
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        query_embedding_str = str(query_embedding)
        return raw_sql_str.replace("[query_vector]", query_embedding_str)


class NLSQLRetriever(BaseRetriever, PromptMixin):
    """
    Text-to-SQL Retriever.

    Retrieves via text.

    Args:
        sql_database (SQLDatabase): SQL database.
        text_to_sql_prompt (BasePromptTemplate): Prompt template for text-to-sql.
            Defaults to DEFAULT_TEXT_TO_SQL_PROMPT.
        context_query_kwargs (dict): Mapping from table name to context query.
            Defaults to None.
        tables (Union[List[str], List[Table]]): List of table names or Table objects.
        table_retriever (ObjectRetriever[SQLTableSchema]): Object retriever for
            SQLTableSchema objects. Defaults to None.
        rows_retriever (Dict[str, VectorIndexRetriever]): a mapping between table name and
            a vector index retriever of its rows. Defaults to None.
        context_str_prefix (str): Prefix for context string. Defaults to None.
        return_raw (bool): Whether to return plain-text dump of SQL results, or parsed into Nodes.
        handle_sql_errors (bool): Whether to handle SQL errors. Defaults to True.
        sql_only (bool) : Whether to get only sql and not the sql query result.
            Default to False.
        llm (Optional[LLM]): Language model to use.

    """

    def __init__(
        self,
        sql_database: SQLDatabase,
        text_to_sql_prompt: Optional[BasePromptTemplate] = None,
        context_query_kwargs: Optional[dict] = None,
        tables: Optional[Union[List[str], List[Table]]] = None,
        table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
        rows_retrievers: Optional[dict[str, BaseRetriever]] = None,
        context_str_prefix: Optional[str] = None,
        sql_parser_mode: SQLParserMode = SQLParserMode.DEFAULT,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        return_raw: bool = True,
        handle_sql_errors: bool = True,
        sql_only: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._sql_retriever = SQLRetriever(sql_database, return_raw=return_raw)
        self._sql_database = sql_database
        self._get_tables = self._load_get_tables_fn(
            sql_database, tables, context_query_kwargs, table_retriever
        )
        self._context_str_prefix = context_str_prefix
        self._llm = llm or Settings.llm
        self._text_to_sql_prompt = text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
        self._sql_parser_mode = sql_parser_mode

        embed_model = embed_model or Settings.embed_model
        self._sql_parser = self._load_sql_parser(sql_parser_mode, embed_model)
        self._handle_sql_errors = handle_sql_errors
        self._sql_only = sql_only
        self._verbose = verbose

        # To retrieve relevant rows from each retrieved table
        self._rows_retrievers = rows_retrievers
        super().__init__(
            callback_manager=callback_manager or Settings.callback_manager,
            verbose=verbose,
        )

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "text_to_sql_prompt": self._text_to_sql_prompt,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "text_to_sql_prompt" in prompts:
            self._text_to_sql_prompt = prompts["text_to_sql_prompt"]

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def _load_sql_parser(
        self, sql_parser_mode: SQLParserMode, embed_model: BaseEmbedding
    ) -> BaseSQLParser:
        """Load SQL parser."""
        if sql_parser_mode == SQLParserMode.DEFAULT:
            return DefaultSQLParser()
        elif sql_parser_mode == SQLParserMode.PGVECTOR:
            return PGVectorSQLParser(embed_model=embed_model)
        else:
            raise ValueError(f"Unknown SQL parser mode: {sql_parser_mode}")

    def _load_get_tables_fn(
        self,
        sql_database: SQLDatabase,
        tables: Optional[Union[List[str], List[Table]]] = None,
        context_query_kwargs: Optional[dict] = None,
        table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
    ) -> Callable[[str], List[SQLTableSchema]]:
        """Load get_tables function."""
        context_query_kwargs = context_query_kwargs or {}
        if table_retriever is not None:
            return lambda query_str: cast(Any, table_retriever).retrieve(query_str)
        else:
            if tables is not None:
                table_names: List[str] = [
                    t.name if isinstance(t, Table) else t for t in tables
                ]
            else:
                table_names = list(sql_database.get_usable_table_names())
            context_strs = [context_query_kwargs.get(t, None) for t in table_names]
            table_schemas = [
                SQLTableSchema(table_name=t, context_str=c)
                for t, c in zip(table_names, context_strs)
            ]
            return lambda _: table_schemas

    def retrieve_with_metadata(
        self, str_or_query_bundle: QueryType
    ) -> Tuple[List[NodeWithScore], Dict]:
        """Retrieve with metadata."""
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        table_desc_str = self._get_table_context(query_bundle)
        logger.info(f"> Table desc str: {table_desc_str}")
        if self._verbose:
            print(f"> Table desc str: {table_desc_str}")

        response_str = self._llm.predict(
            self._text_to_sql_prompt,
            query_str=query_bundle.query_str,
            schema=table_desc_str,
            dialect=self._sql_database.dialect,
        )

        sql_query_str = self._sql_parser.parse_response_to_sql(
            response_str, query_bundle
        )
        # assume that it's a valid SQL query
        logger.debug(f"> Predicted SQL query: {sql_query_str}")
        if self._verbose:
            print(f"> Predicted SQL query: {sql_query_str}")

        if self._sql_only:
            sql_only_node = TextNode(text=f"{sql_query_str}")
            retrieved_nodes = [NodeWithScore(node=sql_only_node)]
            metadata = {"result": sql_query_str}
        else:
            try:
                retrieved_nodes, metadata = self._sql_retriever.retrieve_with_metadata(
                    sql_query_str
                )
            except BaseException as e:
                # if handle_sql_errors is True, then return error message
                if self._handle_sql_errors:
                    err_node = TextNode(text=f"Error: {e!s}")
                    retrieved_nodes = [NodeWithScore(node=err_node)]
                    metadata = {}
                else:
                    raise

        return retrieved_nodes, {"sql_query": sql_query_str, **metadata}

    async def aretrieve_with_metadata(
        self, str_or_query_bundle: QueryType
    ) -> Tuple[List[NodeWithScore], Dict]:
        """Async retrieve with metadata."""
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        table_desc_str = self._get_table_context(query_bundle)
        logger.info(f"> Table desc str: {table_desc_str}")

        response_str = await self._llm.apredict(
            self._text_to_sql_prompt,
            query_str=query_bundle.query_str,
            schema=table_desc_str,
            dialect=self._sql_database.dialect,
        )

        sql_query_str = self._sql_parser.parse_response_to_sql(
            response_str, query_bundle
        )
        # assume that it's a valid SQL query
        logger.debug(f"> Predicted SQL query: {sql_query_str}")

        if self._sql_only:
            sql_only_node = TextNode(text=f"{sql_query_str}")
            retrieved_nodes = [NodeWithScore(node=sql_only_node)]
            metadata: Dict[str, Any] = {}
        else:
            try:
                (
                    retrieved_nodes,
                    metadata,
                ) = await self._sql_retriever.aretrieve_with_metadata(sql_query_str)
            except BaseException as e:
                # if handle_sql_errors is True, then return error message
                if self._handle_sql_errors:
                    err_node = TextNode(text=f"Error: {e!s}")
                    retrieved_nodes = [NodeWithScore(node=err_node)]
                    metadata = {}
                else:
                    raise
        return retrieved_nodes, {"sql_query": sql_query_str, **metadata}

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        retrieved_nodes, _ = self.retrieve_with_metadata(query_bundle)
        return retrieved_nodes

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Async retrieve nodes given query."""
        retrieved_nodes, _ = await self.aretrieve_with_metadata(query_bundle)
        return retrieved_nodes

    def _get_table_context(self, query_bundle: QueryBundle) -> str:
        """Get table context string."""
        table_schema_objs = self._get_tables(query_bundle.query_str)
        context_strs = []
        for table_schema_obj in table_schema_objs:
            # first append table info + additional context
            table_info = self._sql_database.get_single_table_info(
                table_schema_obj.table_name
            )
            if table_schema_obj.context_str:
                table_opt_context = " The table description is: "
                table_opt_context += table_schema_obj.context_str
                table_info += table_opt_context

            # also lookup vector index to return relevant table rows
            # if rows_retrievers was not passed, no rows will be returned
            if self._rows_retrievers is not None:
                rows_retriever = self._rows_retrievers[table_schema_obj.table_name]
                relevant_nodes = rows_retriever.retrieve(query_bundle.query_str)
                if len(relevant_nodes) > 0:
                    table_row_context = "\nHere are some relevant example rows (values in the same order as columns above)\n"
                    for node in relevant_nodes:
                        table_row_context += str(node.get_content()) + "\n"
                    table_info += table_row_context

            if self._verbose:
                print(f"> Table Info: {table_info}")
            context_strs.append(table_info)

        return "\n\n".join(context_strs)
