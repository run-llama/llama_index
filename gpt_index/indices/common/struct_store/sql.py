"""SQL StructDatapointExtractor."""

from gpt_index.indices.common.struct_store.base import BaseStructDatapointExtractor
from gpt_index.data_structs.table import BaseStructTable, StructDatapoint
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TextSplitter
from typing import Dict, Any, cast, Optional
from gpt_index.prompts.prompts import SchemaExtractPrompt
from gpt_index.langchain_helpers.sql_wrapper import SQLDatabase
from gpt_index.schema import BaseDocument
from gpt_index.utils import truncate_text


class SQLStructDatapointExtractor(BaseStructDatapointExtractor):
    """Extracts datapoints from a structured document for a SQL db."""

    def __init__(
        self,
        llm_predictor: LLMPredictor,
        text_splitter: TextSplitter,
        schema_extract_prompt: SchemaExtractPrompt,
        sql_database: SQLDatabase,
        table_name: Optional[str] = None,
        table: Optional[Table] = None,
    ) -> None:
        """Initialize params."""
        super().__init__(llm_predictor, text_splitter, schema_extract_prompt)
        self._llm_predictor = llm_predictor
        self._text_splitter = text_splitter
        self._schema_extract_prompt = schema_extract_prompt
        self._sql_database = sql_database
        self._table_name = table_name

    def _get_col_types_map(self) -> Dict[str, type]:
        """Get col types map for schema."""
        return self._col_types_map

    def _get_schema_text(self) -> str:
        """Insert datapoint into index."""
        return self._sql_database.get_single_table_info(self.table_name)

    def _insert_datapoint(self, datapoint: StructDatapoint) -> None:
        """Insert datapoint into index."""
        datapoint_dict = datapoint.to_dict()["fields"]
        self._sql_database.insert_into_table(
            self._table_name, cast(Dict[Any, Any], datapoint_dict)
        )
