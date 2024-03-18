"""SQL Structured Store."""
from collections import defaultdict
from enum import Enum
from typing import Any, Optional, Sequence, Union

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.data_structs.table import SQLStructTable
from llama_index.core.indices.common.struct_store.schema import SQLContextContainer
from llama_index.core.indices.common.struct_store.sql import (
    SQLStructDatapointExtractor,
)
from llama_index.core.indices.struct_store.base import BaseStructStoreIndex
from llama_index.core.indices.struct_store.container_builder import (
    SQLContextContainerBuilder,
)
from llama_index.core.llms.utils import LLMType
from llama_index.core.schema import BaseNode
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import Settings, llm_from_settings_or_context
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from sqlalchemy import Table


class SQLQueryMode(str, Enum):
    SQL = "sql"
    NL = "nl"


class SQLStructStoreIndex(BaseStructStoreIndex[SQLStructTable]):
    """SQL Struct Store Index.

    The SQLStructStoreIndex is an index that uses a SQL database
    under the hood. During index construction, the data can be inferred
    from unstructured documents given a schema extract prompt,
    or it can be pre-loaded in the database.

    During query time, the user can either specify a raw SQL query
    or a natural language query to retrieve their data.

    NOTE: this is deprecated.

    Args:
        documents (Optional[Sequence[DOCUMENTS_INPUT]]): Documents to index.
            NOTE: in the SQL index, this is an optional field.
        sql_database (Optional[SQLDatabase]): SQL database to use,
            including table names to specify.
            See :ref:`Ref-Struct-Store` for more details.
        table_name (Optional[str]): Name of the table to use
            for extracting data.
            Either table_name or table must be specified.
        table (Optional[Table]): SQLAlchemy Table object to use.
            Specifying the Table object explicitly, instead of
            the table name, allows you to pass in a view.
            Either table_name or table must be specified.
        sql_context_container (Optional[SQLContextContainer]): SQL context container.
            an be generated from a SQLContextContainerBuilder.
            See :ref:`Ref-Struct-Store` for more details.

    """

    index_struct_cls = SQLStructTable

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[SQLStructTable] = None,
        service_context: Optional[ServiceContext] = None,
        sql_database: Optional[SQLDatabase] = None,
        table_name: Optional[str] = None,
        table: Optional[Table] = None,
        ref_doc_id_column: Optional[str] = None,
        sql_context_container: Optional[SQLContextContainer] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if sql_database is None:
            raise ValueError("sql_database must be specified")
        self.sql_database = sql_database
        # needed here for data extractor
        self._ref_doc_id_column = ref_doc_id_column
        self._table_name = table_name
        self._table = table

        # if documents aren't specified, pass in a blank []
        if index_struct is None:
            nodes = nodes or []

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            **kwargs,
        )

        # TODO: index_struct context_dict is deprecated,
        # we're migrating storage of information to here.
        if sql_context_container is None:
            container_builder = SQLContextContainerBuilder(sql_database)
            sql_context_container = container_builder.build_context_container()
        self.sql_context_container = sql_context_container

    @property
    def ref_doc_id_column(self) -> Optional[str]:
        return self._ref_doc_id_column

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> SQLStructTable:
        """Build index from nodes."""
        index_struct = self.index_struct_cls()
        if len(nodes) == 0:
            return index_struct
        else:
            data_extractor = SQLStructDatapointExtractor(
                llm_from_settings_or_context(Settings, self.service_context),
                self.schema_extract_prompt,
                self.output_parser,
                self.sql_database,
                table_name=self._table_name,
                table=self._table,
                ref_doc_id_column=self._ref_doc_id_column,
            )
            # group nodes by ids
            source_to_node = defaultdict(list)
            for node in nodes:
                source_to_node[node.ref_doc_id].append(node)

            for node_set in source_to_node.values():
                data_extractor.insert_datapoint_from_nodes(node_set)
        return index_struct

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a document."""
        data_extractor = SQLStructDatapointExtractor(
            llm_from_settings_or_context(Settings, self._service_context),
            self.schema_extract_prompt,
            self.output_parser,
            self.sql_database,
            table_name=self._table_name,
            table=self._table,
            ref_doc_id_column=self._ref_doc_id_column,
        )
        data_extractor.insert_datapoint_from_nodes(nodes)

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        raise NotImplementedError("Not supported")

    def as_query_engine(
        self,
        llm: Optional[LLMType] = None,
        query_mode: Union[str, SQLQueryMode] = SQLQueryMode.NL,
        **kwargs: Any,
    ) -> BaseQueryEngine:
        # NOTE: lazy import
        from llama_index.core.indices.struct_store.sql_query import (
            NLStructStoreQueryEngine,
            SQLStructStoreQueryEngine,
        )

        if query_mode == SQLQueryMode.NL:
            return NLStructStoreQueryEngine(self, **kwargs)
        elif query_mode == SQLQueryMode.SQL:
            return SQLStructStoreQueryEngine(self, **kwargs)
        else:
            raise ValueError(f"Unknown query mode: {query_mode}")


GPTSQLStructStoreIndex = SQLStructStoreIndex
