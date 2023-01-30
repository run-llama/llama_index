"""SQLite structured store."""

from typing import Any, Dict, List, Optional, Sequence, Type, cast

from sqlalchemy import Table

from gpt_index.data_structs.table import SQLStructTable, StructDatapoint
from gpt_index.indices.base import DOCUMENTS_INPUT
from gpt_index.indices.common.struct_store.base import SQLContextBuilder
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.query.struct_store.sql import (
    GPTNLStructStoreIndexQuery,
    GPTSQLStructStoreIndexQuery,
)
from gpt_index.indices.struct_store.base import BaseGPTStructStoreIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.sql_wrapper import SQLDatabase
from gpt_index.schema import BaseDocument


class GPTSQLStructStoreIndex(BaseGPTStructStoreIndex[SQLStructTable]):
    """Base GPT SQL Struct Store Index.

    The GPTSQLStructStoreIndex is an index that uses a SQL database
    under the hood. During index construction, the data can be inferred
    from unstructured documents given a schema extract prompt,
    or it can be pre-loaded in the database.

    During query time, the user can either specify a raw SQL query
    or a natural language query to retrieve their data.

    Args:
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
        table_context_dict (Optional[Dict[str, str]]): Optional table context to use.
            If specified,
            sql_context_builder and context_documents cannot be specified.
        sql_context_builder (Optional[SQLContextBuilder]): SQL context builder.
            If specified, the context builder will be used to build
            context for the specified table, which will then be used during
            query-time. Also if specified, context_documents must be specified,
            and table_context cannot be specified.
        context_documents_dict (Optional[Dict[str, List[BaseDocument]]]):
            Optional context
            documents to inform the sql_context_builder. Must be specified if
            sql_context_builder is specified. Cannot be specified if table_context
            is specified.

    """

    index_struct_cls = SQLStructTable

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[SQLStructTable] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        sql_database: Optional[SQLDatabase] = None,
        table_name: Optional[str] = None,
        table: Optional[Table] = None,
        ref_doc_id_column: Optional[str] = None,
        table_context_dict: Optional[Dict[str, str]] = None,
        sql_context_builder: Optional[SQLContextBuilder] = None,
        context_documents_dict: Optional[Dict[str, List[BaseDocument]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # currently the user must specify a table info
        if table_name is None and table is None:
            raise ValueError("table_name must be specified")
        self.table_name = table_name or cast(Table, table).name
        if sql_database is None:
            raise ValueError("sql_database must be specified")
        self.sql_database = sql_database
        if table is None:
            table = self.sql_database.metadata_obj.tables[table_name]
        # if ref_doc_id_column is specified, then we need to check that
        # it is a valid column in the table
        col_names = [c.name for c in table.c]
        if ref_doc_id_column is not None and ref_doc_id_column not in col_names:
            raise ValueError(
                f"ref_doc_id_column {ref_doc_id_column} not in table {table_name}"
            )
        self.ref_doc_id_column = ref_doc_id_column
        # then store python types of each column
        self._col_types_map: Dict[str, type] = {
            c.name: table.c[c.name].type.python_type for c in table.c
        }

        super().__init__(
            documents=documents,
            index_struct=index_struct,
            llm_predictor=llm_predictor,
            **kwargs,
        )

        # if context builder is specified, then add to context_dict
        if table_context_dict is not None and (
            sql_context_builder is not None or context_documents_dict is not None
        ):
            raise ValueError(
                "Cannot specify both table_context_dict and "
                "sql_context_builder/context_documents_dict"
            )
        if sql_context_builder is not None:
            if context_documents_dict is None:
                raise ValueError(
                    "context_documents_dict must be specified if "
                    "sql_context_builder is specified"
                )
            context_documents_dict = cast(
                Dict[str, List[BaseDocument]], context_documents_dict
            )
            context_dict: Dict[
                str, str
            ] = sql_context_builder.build_all_context_from_documents(
                context_documents_dict
            )
        elif table_context_dict is not None:
            context_dict = table_context_dict
        else:
            context_dict = {}

        # validate context_dict keys are valid table names
        context_keys = set(context_dict.keys())
        if not context_keys.issubset(set(self.sql_database.get_table_names())):
            raise ValueError(
                "Invalid context table names: "
                f"{context_keys - set(self.sql_database.get_table_names())}"
            )

        self._index_struct.context_dict.update(context_dict)
        self._sql_context_builder = sql_context_builder

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTNLStructStoreIndexQuery,
            QueryMode.SQL: GPTSQLStructStoreIndexQuery,
        }

    def _get_col_types_map(self) -> Dict[str, type]:
        """Get col types map for schema."""
        return self._col_types_map

    def _get_schema_text(self) -> str:
        """Insert datapoint into index."""
        return self.sql_database.get_single_table_info(self.table_name)

    def _insert_datapoint(self, datapoint: StructDatapoint) -> None:
        """Insert datapoint into index."""
        datapoint_dict = datapoint.to_dict()["fields"]
        self.sql_database.insert_into_table(
            self.table_name, cast(Dict[Any, Any], datapoint_dict)
        )

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        """Preprocess query.

        This allows subclasses to pass in additional query kwargs
        to query, for instance arguments that are shared between the
        index and the query class. By default, this does nothing.
        This also allows subclasses to do validation.

        """
        super()._preprocess_query(mode, query_kwargs)
        # pass along sql_database, table_name
        query_kwargs["sql_database"] = self.sql_database
        if mode == QueryMode.DEFAULT:
            query_kwargs["ref_doc_id_column"] = self.ref_doc_id_column
