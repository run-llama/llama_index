"""SQLite structured store."""

from typing import Any, Generic, Optional, Sequence, TypeVar, Callable

from gpt_index.data_structs.table import SQLiteStructTable
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.base import DOCUMENTS_INPUT, BaseGPTIndex
from gpt_index.data_structs.table import StructDatapoint, StructValue
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.default_prompts import DEFAULT_SCHEMA_EXTRACT_PROMPT
from gpt_index.prompts.prompts import SchemaExtractPrompt
from gpt_index.schema import BaseDocument
from gpt_index.indices.utils import truncate_text
from gpt_index.indices.struct_store.base import BaseGPTStructStoreIndex
from gpt_index.langchain_helpers.sql_wrapper import SQLDatabase
from gpt_index.indices.query.schema import QueryMode
from sqlalchemy import create_engine, inspect


class GPTSQLStructStoreIndex(BaseGPTStructStoreIndex[SQLiteStructTable]):
    """Base GPT SQL Struct Store Index.

    Uses a SQLAlchemy wrapper to connect to underlying storage.
    
    """

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[SQLiteStructTable] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        sql_database: Optional[SQLDatabase] = None,
        table_name: Optional[str] = None,
        ref_doc_id_column: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # currently the user must specify a table info
        self.table_name = table_name
        self.sql_database = sql_database
        # if ref_doc_id_column is specified, then we need to check that 
        # it is a valid column in the table
        columns = self.sql_database.get_table_columns(table_name)
        col_names = [col['name'] for col in columns]
        if ref_doc_id_column is not None and ref_doc_id_column not in col_names:
                raise ValueError(f"ref_doc_id_column {ref_doc_id_column} not in table {table_name}")
        self.ref_doc_id_column = ref_doc_id_column
        
        super().__init__(
            documents=documents,
            index_struct=index_struct,
            llm_predictor=llm_predictor,
            **kwargs,
        )

    def _get_schema_text(self) -> str:
        """Insert datapoint into index."""
        return self.sql_database.get_single_table_info(self.table_name)

    def _insert_datapoint(self, datapoint: StructDatapoint) -> None:
        """Insert datapoint into index."""
        datapoint_dict = datapoint.fields.to_dict()
        self.sql_database.insert_into_table(self.table_name, datapoint_dict)

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