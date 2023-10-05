"""SQL Container builder."""


from typing import Any, Dict, List, Optional, Type

from llama_index.indices.base import BaseIndex
from llama_index.indices.common.struct_store.base import SQLDocumentContextBuilder
from llama_index.indices.common.struct_store.schema import SQLContextContainer
from llama_index.indices.query.schema import QueryType
from llama_index.readers.base import Document
from llama_index.schema import BaseNode
from llama_index.utilities.sql_wrapper import SQLDatabase

DEFAULT_CONTEXT_QUERY_TMPL = (
    "Please return the relevant tables (including the full schema) "
    "for the following query: {orig_query_str}"
)


class SQLContextContainerBuilder:
    """SQLContextContainerBuilder.

    Build a SQLContextContainer that can be passed to the SQL index
    during index construction or during query-time.

    NOTE: if context_str is specified, that will be used as context
    instead of context_dict

    Args:
        sql_database (SQLDatabase): SQL database
        context_dict (Optional[Dict[str, str]]): context dict

    """

    def __init__(
        self,
        sql_database: SQLDatabase,
        context_dict: Optional[Dict[str, str]] = None,
        context_str: Optional[str] = None,
    ):
        """Initialize params."""
        self.sql_database = sql_database

        # if context_dict provided, validate that all keys are valid table names
        if context_dict is not None:
            # validate context_dict keys are valid table names
            context_keys = set(context_dict.keys())
            if not context_keys.issubset(
                set(self.sql_database.get_usable_table_names())
            ):
                raise ValueError(
                    "Invalid context table names: "
                    f"{context_keys - set(self.sql_database.get_usable_table_names())}"
                )
        self.context_dict = context_dict or {}
        # build full context from sql_database
        self.full_context_dict = self._build_context_from_sql_database(
            self.sql_database, current_context=self.context_dict
        )
        self.context_str = context_str

    @classmethod
    def from_documents(
        cls,
        documents_dict: Dict[str, List[BaseNode]],
        sql_database: SQLDatabase,
        **context_builder_kwargs: Any,
    ) -> "SQLContextContainerBuilder":
        """Build context from documents."""
        context_builder = SQLDocumentContextBuilder(
            sql_database, **context_builder_kwargs
        )
        context_dict = context_builder.build_all_context_from_documents(documents_dict)
        return SQLContextContainerBuilder(sql_database, context_dict=context_dict)

    def _build_context_from_sql_database(
        self,
        sql_database: SQLDatabase,
        current_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Get tables schema + optional context as a single string."""
        current_context = current_context or {}
        result_context = {}
        for table_name in sql_database.get_usable_table_names():
            table_desc = sql_database.get_single_table_info(table_name)
            table_text = f"Schema of table {table_name}:\n" f"{table_desc}\n"
            if table_name in current_context:
                table_text += f"Context of table {table_name}:\n"
                table_text += current_context[table_name]
            result_context[table_name] = table_text
        return result_context

    def _get_context_dict(self, ignore_db_schema: bool) -> Dict[str, str]:
        """Get full context dict."""
        if ignore_db_schema:
            return self.context_dict
        else:
            return self.full_context_dict

    def derive_index_from_context(
        self,
        index_cls: Type[BaseIndex],
        ignore_db_schema: bool = False,
        **index_kwargs: Any,
    ) -> BaseIndex:
        """Derive index from context."""
        full_context_dict = self._get_context_dict(ignore_db_schema)
        context_docs = []
        for table_name, context_str in full_context_dict.items():
            doc = Document(text=context_str, metadata={"table_name": table_name})
            context_docs.append(doc)
        return index_cls.from_documents(
            documents=context_docs,
            **index_kwargs,
        )

    def query_index_for_context(
        self,
        index: BaseIndex,
        query_str: QueryType,
        query_tmpl: Optional[str] = DEFAULT_CONTEXT_QUERY_TMPL,
        store_context_str: bool = True,
        **index_kwargs: Any,
    ) -> str:
        """Query index for context.

        A simple wrapper around the index.query call which
        injects a query template to specifically fetch table information,
        and can store a context_str.

        Args:
            index (BaseIndex): index data structure
            query_str (QueryType): query string
            query_tmpl (Optional[str]): query template
            store_context_str (bool): store context_str

        """
        if query_tmpl is None:
            context_query_str = query_str
        else:
            context_query_str = query_tmpl.format(orig_query_str=query_str)
        query_engine = index.as_query_engine()
        response = query_engine.query(context_query_str)
        context_str = str(response)
        if store_context_str:
            self.context_str = context_str
        return context_str

    def build_context_container(
        self, ignore_db_schema: bool = False
    ) -> SQLContextContainer:
        """Build index structure."""
        full_context_dict = self._get_context_dict(ignore_db_schema)
        return SQLContextContainer(
            context_str=self.context_str,
            context_dict=full_context_dict,
        )
