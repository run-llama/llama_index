import logging
from typing import Any, List, Optional, Union

from sqlalchemy import Table

from llama_index.data_structs.node import NodeWithScore
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.struct_store.sql_table import SQLTableIndex

DEFAULT_QUERY_TMPL = (
    "Please return the relevant tables (including the full schema) "
    "for the following query: {orig_query_str}"
)

logger = logging.getLogger(__name__)


class SQLTableRetriever(BaseRetriever):
    """Retriever for SQL Table Index."""

    def __init__(
        self,
        index: SQLTableIndex,
        query_tmpl: Optional[str] = None,
        tables: Optional[List[Union[str, Table]]] = None,
        **kwargs: Any
    ) -> None:
        self._index = index
        self._query_tmpl = query_tmpl or DEFAULT_QUERY_TMPL
        self._tables = tables

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Retrieve nodes."""
        index = self._index.index

        # Specific table case. Use initialized tables if specified.
        if self._tables:
            vector_index_retriever = index.as_retriever(similarity_top_k=1)
            tables = [
                str(table.name) if isinstance(table, Table) else table
                for table in self._tables
            ]
            scored_nodes: List[NodeWithScore] = []
            embed_model = self._index.service_context.embed_model
            for table in tables:
                new_query = QueryBundle(
                    "table_name: " + table,
                    embedding=embed_model.get_query_embedding("table_name: " + table),
                )
                result = vector_index_retriever.retrieve(new_query)
                scored_nodes.extend(result)
            logger.debug("Retrieved nodes: %s", scored_nodes)
            return scored_nodes
        else:
            context_query_str = self._query_tmpl.format(
                orig_query_str=query_bundle.query_str
            )
            vector_index_retriever = index.as_retriever()
            return vector_index_retriever.retrieve(QueryBundle(context_query_str))
