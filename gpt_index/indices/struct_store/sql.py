"""SQLite structured store."""

from typing import Any, Dict, Optional, Sequence, cast

from sqlalchemy import Table
from sqlalchemy.engine import Engine

from gpt_index.data_structs.table import SQLStructTable, StructDatapoint
from gpt_index.indices.base import DOCUMENTS_INPUT
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.struct_store.base import BaseGPTStructStoreIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.sql_wrapper import SQLDatabase


class GPTSQLStructStoreIndex(BaseGPTStructStoreIndex[SQLStructTable]):
    """Base GPT SQL Struct Store Index.

    The GPTSQLStructStoreIndex is an index that uses a SQL database
    under the hood. During index construction, the data can be inferred
    from unstructured  documents given a schema extract prompt,
    or it can be pre-loaded in the database.

    During query time, the user can either specify a raw SQL query
    or a natural language query to retrieve their data.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.

    """

    index_struct_cls = SQLStructTable

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[SQLStructTable] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        sql_engine: Optional[Engine] = None,
        table_name: Optional[str] = None,
        table: Optional[Table] = None,
        ref_doc_id_column: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # currently the user must specify a table info
        if table_name is None and table is None:
            raise ValueError("table_name must be specified")
        self.table_name = table_name or cast(Table, table).name
        if sql_engine is None:
            raise ValueError("sql_engine must be specified")
        self.sql_database = SQLDatabase(sql_engine)
        # if ref_doc_id_column is specified, then we need to check that
        # it is a valid column in the table
        if table is None:
            table = self.sql_database.metadata_obj.tables[table_name]
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
        query_kwargs["table_name"] = self.table_name
        query_kwargs["ref_doc_id_column"] = self.ref_doc_id_column
