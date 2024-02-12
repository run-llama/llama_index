"""SQL StructDatapointExtractor."""

from typing import Any, Dict, Optional, cast

from llama_index.core.data_structs.table import StructDatapoint
from llama_index.core.indices.common.struct_store.base import (
    OUTPUT_PARSER_TYPE,
    BaseStructDatapointExtractor,
)
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.service_context_elements.llm_predictor import (
    LLMPredictorType,
)
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from sqlalchemy import Table


class SQLStructDatapointExtractor(BaseStructDatapointExtractor):
    """Extracts datapoints from a structured document for a SQL db."""

    def __init__(
        self,
        llm: LLMPredictorType,
        schema_extract_prompt: BasePromptTemplate,
        output_parser: OUTPUT_PARSER_TYPE,
        sql_database: SQLDatabase,
        table_name: Optional[str] = None,
        table: Optional[Table] = None,
        ref_doc_id_column: Optional[str] = None,
    ) -> None:
        """Initialize params."""
        super().__init__(llm, schema_extract_prompt, output_parser)
        self._sql_database = sql_database
        # currently the user must specify a table info
        if table_name is None and table is None:
            raise ValueError("table_name must be specified")
        self._table_name = table_name or cast(Table, table).name
        if table is None:
            table_name = cast(str, table_name)
            table = self._sql_database.metadata_obj.tables[table_name]
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

    def _get_col_types_map(self) -> Dict[str, type]:
        """Get col types map for schema."""
        return self._col_types_map

    def _get_schema_text(self) -> str:
        """Insert datapoint into index."""
        return self._sql_database.get_single_table_info(self._table_name)

    def _insert_datapoint(self, datapoint: StructDatapoint) -> None:
        """Insert datapoint into index."""
        datapoint_dict = datapoint.to_dict()["fields"]
        self._sql_database.insert_into_table(
            self._table_name, cast(Dict[Any, Any], datapoint_dict)
        )
