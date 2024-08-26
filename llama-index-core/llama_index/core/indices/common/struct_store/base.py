"""Common classes for structured operations."""

import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, cast

from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.data_structs.table import StructDatapoint
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.node_parser.interface import TextSplitter
from llama_index.core.llms import LLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_REFINE_TABLE_CONTEXT_PROMPT_SEL,
)
from llama_index.core.prompts.default_prompts import (
    DEFAULT_TABLE_CONTEXT_PROMPT,
    DEFAULT_TABLE_CONTEXT_QUERY,
)
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.settings import Settings
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.utils import truncate_text

logger = logging.getLogger(__name__)


class SQLDocumentContextBuilder:
    """Builder that builds context for a given set of SQL tables.

    Args:
        sql_database (Optional[SQLDatabase]): SQL database to use,
        text_splitter (Optional[TextSplitter]): Text Splitter to use.
        table_context_prompt (Optional[BasePromptTemplate]): A
            Table Context Prompt (see :ref:`Prompt-Templates`).
        refine_table_context_prompt (Optional[BasePromptTemplate]):
            A Refine Table Context Prompt (see :ref:`Prompt-Templates`).
        table_context_task (Optional[str]): The query to perform
            on the table context. A default query string is used
            if none is provided by the user.
    """

    def __init__(
        self,
        sql_database: SQLDatabase,
        llm: Optional[LLM] = None,
        text_splitter: Optional[TextSplitter] = None,
        table_context_prompt: Optional[BasePromptTemplate] = None,
        refine_table_context_prompt: Optional[BasePromptTemplate] = None,
        table_context_task: Optional[str] = None,
    ) -> None:
        """Initialize params."""
        # TODO: take in an entire index instead of forming a response builder
        if sql_database is None:
            raise ValueError("sql_database must be provided.")
        self._sql_database = sql_database
        self._text_splitter = text_splitter
        self._llm = llm or Settings.llm
        self._prompt_helper = Settings._prompt_helper or PromptHelper.from_llm_metadata(
            self._llm.metadata,
        )
        self._callback_manager = Settings.callback_manager
        self._table_context_prompt = (
            table_context_prompt or DEFAULT_TABLE_CONTEXT_PROMPT
        )
        self._refine_table_context_prompt = (
            refine_table_context_prompt or DEFAULT_REFINE_TABLE_CONTEXT_PROMPT_SEL
        )
        self._table_context_task = table_context_task or DEFAULT_TABLE_CONTEXT_QUERY

    def build_all_context_from_documents(
        self,
        documents_dict: Dict[str, List[BaseNode]],
    ) -> Dict[str, str]:
        """Build context for all tables in the database."""
        context_dict = {}
        for table_name in self._sql_database.get_usable_table_names():
            context_dict[table_name] = self.build_table_context_from_documents(
                documents_dict[table_name], table_name
            )
        return context_dict

    def build_table_context_from_documents(
        self,
        documents: Sequence[BaseNode],
        table_name: str,
    ) -> str:
        """Build context from documents for a single table."""
        schema = self._sql_database.get_single_table_info(table_name)
        prompt_with_schema = self._table_context_prompt.partial_format(schema=schema)
        prompt_with_schema.metadata["prompt_type"] = PromptType.QUESTION_ANSWER
        refine_prompt_with_schema = self._refine_table_context_prompt.partial_format(
            schema=schema
        )
        refine_prompt_with_schema.metadata["prompt_type"] = PromptType.REFINE

        text_splitter = (
            self._text_splitter
            or self._prompt_helper.get_text_splitter_given_prompt(prompt_with_schema)
        )
        # we use the ResponseBuilder to iteratively go through all texts
        response_builder = get_response_synthesizer(
            llm=self._llm,
            text_qa_template=prompt_with_schema,
            refine_template=refine_prompt_with_schema,
        )
        with self._callback_manager.event(
            CBEventType.CHUNKING,
            payload={EventPayload.DOCUMENTS: documents},
        ) as event:
            text_chunks = []
            for doc in documents:
                chunks = text_splitter.split_text(
                    doc.get_content(metadata_mode=MetadataMode.LLM)
                )
                text_chunks.extend(chunks)

            event.on_end(
                payload={EventPayload.CHUNKS: text_chunks},
            )

        # feed in the "query_str" or the task
        table_context = response_builder.get_response(
            text_chunks=text_chunks, query_str=self._table_context_task
        )
        return cast(str, table_context)


OUTPUT_PARSER_TYPE = Callable[[str], Optional[Dict[str, Any]]]


class BaseStructDatapointExtractor:
    """Extracts datapoints from a structured document."""

    def __init__(
        self,
        llm: LLM,
        schema_extract_prompt: BasePromptTemplate,
        output_parser: OUTPUT_PARSER_TYPE,
    ) -> None:
        """Initialize params."""
        self._llm = llm
        self._schema_extract_prompt = schema_extract_prompt
        self._output_parser = output_parser

    def _clean_and_validate_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Validate fields with col_types_map."""
        new_fields = {}
        col_types_map = self._get_col_types_map()
        for field, value in fields.items():
            clean_value = value
            if field not in col_types_map:
                continue
            # if expected type is int or float, try to convert value to int or float
            expected_type = col_types_map[field]
            if expected_type == int:
                try:
                    clean_value = int(value)
                except ValueError:
                    continue
            elif expected_type == float:
                try:
                    clean_value = float(value)
                except ValueError:
                    continue
            else:
                if len(value) == 0:
                    continue
                if not isinstance(value, col_types_map[field]):
                    continue
            new_fields[field] = clean_value
        return new_fields

    @abstractmethod
    def _insert_datapoint(self, datapoint: StructDatapoint) -> None:
        """Insert datapoint into index."""

    @abstractmethod
    def _get_col_types_map(self) -> Dict[str, type]:
        """Get col types map for schema."""

    @abstractmethod
    def _get_schema_text(self) -> str:
        """Get schema text for extracting relevant info from unstructured text."""

    def insert_datapoint_from_nodes(self, nodes: Sequence[BaseNode]) -> None:
        """Extract datapoint from a document and insert it."""
        text_chunks = [
            node.get_content(metadata_mode=MetadataMode.LLM) for node in nodes
        ]
        fields = {}
        for i, text_chunk in enumerate(text_chunks):
            fmt_text_chunk = truncate_text(text_chunk, 50)
            logger.info(f"> Adding chunk {i}: {fmt_text_chunk}")
            # if embedding specified in document, pass it to the Node
            schema_text = self._get_schema_text()
            response_str = self._llm.predict(
                self._schema_extract_prompt,
                text=text_chunk,
                schema=schema_text,
            )
            cur_fields = self._output_parser(response_str)
            if cur_fields is None:
                continue
            # validate fields with col_types_map
            new_cur_fields = self._clean_and_validate_fields(cur_fields)
            fields.update(new_cur_fields)
        struct_datapoint = StructDatapoint(fields)
        if struct_datapoint is not None:
            self._insert_datapoint(struct_datapoint)
            logger.debug(f"> Added datapoint: {fields}")
