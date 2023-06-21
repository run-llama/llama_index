"""SQL Structured Store."""
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Union

from llama_index.data_structs.node import Node
from llama_index.data_structs.table import SQLStructTable
from llama_index.indices.base import BaseIndex
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.struct_store.base import BaseStructStoreIndex
from llama_index.langchain_helpers.sql_wrapper import SQLDatabase


class SQLQueryMode(str, Enum):
    SQL = "sql"
    NL = "nl"


class SQLTableIndex(BaseStructStoreIndex[SQLStructTable]):
    """SQL Table Index.

    The SQLTableIndex is an index that references a SQL database table schema.
    During index construction, the user can

    During query time, the user can either specify a raw SQL query
    or a natural language query to retrieve their data.

    Args:
        sql_database (SQLDatabase): SQL database to use,
            including table names to specify.
            See :ref:`Ref-Struct-Store` for more details.
        vector_index (Optional[VectorStoreIndex]): Vector index to use for
            storing table data.
        service_context (Optional[ServiceContext]): Service context.
        tables (Optional[List[Union[str, Table]]]): List of tables to use.
        extra_context_dict (Optional[Dict[str, str]]): Extra context dict. Keys
            should be table names.
    """

    index_struct_cls = SQLStructTable

    def __init__(
        self,
        sql_database: SQLDatabase,
        index: BaseIndex,
        service_context: Optional[ServiceContext] = None,
        extra_context_dict: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if sql_database is None:
            raise ValueError("sql_database must be specified")
        self.sql_database = sql_database
        self._index = index
        self._service_context = service_context or ServiceContext.from_defaults()
        self._extra_context_dict = extra_context_dict or {}

        super().__init__(
            nodes=[],
            service_context=service_context,
            **kwargs,
        )
        self._build_index_from_tables()

    def _build_index_from_tables(self) -> None:
        """Build index from sql tables."""
        nodes = []
        for table_name in self.sql_database.get_usable_table_names():
            table_desc = self.sql_database.get_single_table_info(table_name)
            table_text = f"Schema of table {table_name}:\n" f"{table_desc}\n"
            nodes.append(Node(text=table_text, extra_info={"table_name": table_name}))
        for table_name, table_context in self._extra_context_dict.items():
            nodes.append(
                Node(
                    text=f"{table_name}: {table_context}",
                    extra_info={"table_name": table_name},
                )
            )
        self._index.insert_nodes(nodes)

    def _build_index_from_nodes(self, nodes: Sequence[Node]) -> SQLStructTable:
        index_struct = self.index_struct_cls()
        return index_struct

    @property
    def index(self) -> BaseIndex:
        return self._index

    def _insert(self, nodes: Sequence[Node], **insert_kwargs: Any) -> None:
        """Insert a document."""

        raise NotImplementedError("Not supported")

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        from llama_index.indices.struct_store.retriever import SQLTableRetriever

        return SQLTableRetriever(self, **kwargs)

    def as_query_engine(
        self, query_mode: Union[str, SQLQueryMode] = SQLQueryMode.NL, **kwargs: Any
    ) -> BaseQueryEngine:
        # NOTE: lazy import
        from llama_index.indices.struct_store.sql_query import (
            NLSQLTableQueryEngine,
            SQLTableQueryEngine,
        )

        if query_mode == SQLQueryMode.NL:
            return NLSQLTableQueryEngine(self, **kwargs)
        elif query_mode == SQLQueryMode.SQL:
            return SQLTableQueryEngine(self, **kwargs)
        else:
            raise ValueError(f"Unknown query mode: {query_mode}")
