"""SQL Structured Store."""
import json
from typing import Any, Dict, Optional, Sequence, Type, cast

from sqlalchemy import Table

import logging
from gpt_index.indices.common.struct_store.sql import SQLStructDatapointExtractor
from gpt_index.utils import truncate_text
from gpt_index.data_structs.table import SQLStructTable, StructDatapoint
from gpt_index.indices.base import DOCUMENTS_INPUT, BaseGPTIndex
from gpt_index.indices.common.struct_store.schema import SQLContextContainer
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.query.struct_store.sql import (
    GPTNLStructStoreIndexQuery,
    GPTSQLStructStoreIndexQuery,
)
from gpt_index.indices.struct_store.base import BaseGPTStructStoreIndex
from gpt_index.indices.struct_store.container_builder import SQLContextContainerBuilder
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.sql_wrapper import SQLDatabase
from gpt_index.schema import BaseDocument
from gpt_index.langchain_helpers.text_splitter import TextSplitter


class GPTSQLStructStoreIndex(BaseGPTStructStoreIndex[SQLStructTable]):
    """Base GPT SQL Struct Store Index.

    The GPTSQLStructStoreIndex is an index that uses a SQL database
    under the hood. During index construction, the data can be inferred
    from unstructured documents given a schema extract prompt,
    or it can be pre-loaded in the database.

    During query time, the user can either specify a raw SQL query
    or a natural language query to retrieve their data.

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
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[SQLStructTable] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        sql_database: Optional[SQLDatabase] = None,
        table_name: Optional[str] = None,
        table: Optional[Table] = None,
        ref_doc_id_column: Optional[str] = None,
        sql_context_container: Optional[SQLContextContainer] = None,
        text_splitter: Optional[TextSplitter] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if sql_database is None:
            raise ValueError("sql_database must be specified")
        self.sql_database = sql_database
        # needed here for data extractor
        self._text_splitter = text_splitter or self._build_fallback_text_splitter()

        # if documents aren't specified, pass in a blank []
        documents = documents or []
        if len(documents) == 0:
            # if there are no documents specified, no need for table_name/table
            pass
        else:
            # currently the user must specify a table info
            if table_name is None and table is None:
                raise ValueError("table_name must be specified")
            self.table_name = table_name or cast(Table, table).name
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
            # TODO: make this exposed externally
            self._data_extractor = SQLStructDatapointExtractor(
                self._llm_predictor,
                self._text_splitter,
                self.schema_extract_prompt,
                self.sql_database,
                self.table_name,
            )

        super().__init__(
            documents=documents,
            index_struct=index_struct,
            llm_predictor=llm_predictor,
            **kwargs,
        )

        # TODO: index_struct context_dict is deprecated,
        # we're migrating storage of information to here.
        if sql_context_container is None:
            container_builder = SQLContextContainerBuilder(sql_database)
            sql_context_container = container_builder.build_context_container()
        self.sql_context_container = sql_context_container

    def _add_document_to_index(
        self,
        document: BaseDocument,
    ) -> None:
        """Add document to index."""
        text_chunks = self._text_splitter.split_text(document.get_text())
        fields = {}
        for i, text_chunk in enumerate(text_chunks):
            fmt_text_chunk = truncate_text(text_chunk, 50)
            logging.info(f"> Adding chunk {i}: {fmt_text_chunk}")
            # if embedding specified in document, pass it to the Node
            schema_text = self._get_schema_text()
            response_str, _ = self._llm_predictor.predict(
                self.schema_extract_prompt,
                text=text_chunk,
                schema=schema_text,
            )
            cur_fields = self.output_parser(response_str)
            if cur_fields is None:
                continue
            # validate fields with col_types_map
            new_cur_fields = self._clean_and_validate_fields(cur_fields)
            fields.update(new_cur_fields)

        struct_datapoint = StructDatapoint(fields)
        if struct_datapoint is not None:
            self._insert_datapoint(struct_datapoint)
            logging.debug(f"> Added datapoint: {fields}")

    def _build_index_from_documents(
        self, documents: Sequence[BaseDocument]
    ) -> SQLStructTable:
        """Build index from documents."""
        index_struct = self.index_struct_cls()
        for d in documents:
            self._add_document_to_index(d)
        return index_struct

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
        if "sql_context_container" not in query_kwargs:
            query_kwargs["sql_context_container"] = self.sql_context_container
        if mode == QueryMode.DEFAULT:
            query_kwargs["ref_doc_id_column"] = self.ref_doc_id_column

    @classmethod
    def load_from_string(cls, index_string: str, **kwargs: Any) -> "BaseGPTIndex":
        """Load index from string (in JSON-format).

        This method loads the index from a JSON string. The index data
        structure itself is preserved completely. If the index is defined over
        subindices, those subindices will also be preserved (and subindices of
        those subindices, etc.).

        NOTE: load_from_string should not be used for indices composed on top
        of other indices. Please define a `ComposableGraph` and use
        `save_to_string` and `load_from_string` on that instead.

        Args:
            index_string (str): The index string (in JSON-format).

        Returns:
            BaseGPTIndex: The loaded index.

        """
        # NOTE: also getting deserialized in parent class,
        # figure out how to deal with later
        result_dict = json.loads(index_string)
        sql_context_container = SQLContextContainer.from_dict(
            result_dict["sql_context_container"]
        )
        result_obj = super().load_from_string(
            index_string, sql_context_container=sql_context_container, **kwargs
        )
        return result_obj

    def save_to_string(self, **save_kwargs: Any) -> str:
        """Save to string.

        This method stores the index into a JSON string.

        NOTE: save_to_string should not be used for indices composed on top
        of other indices. Please define a `ComposableGraph` and use
        `save_to_string` and `load_from_string` on that instead.

        Returns:
            str: The JSON string of the index.

        """
        if self.docstore.contains_index_struct(
            exclude_ids=[self.index_struct.get_doc_id()]
        ):
            raise ValueError(
                "Cannot call `save_to_string` on index if index is composed on top of "
                "other indices. Please define a `ComposableGraph` and use "
                "`save_to_string` and `load_from_string` on that instead."
            )
        out_dict: Dict[str, Any] = {
            "index_struct_id": self.index_struct.get_doc_id(),
            "docstore": self.docstore.serialize_to_dict(),
            "sql_context_container": self.sql_context_container.to_dict(),
        }
        return json.dumps(out_dict, **save_kwargs)
