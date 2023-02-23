"""Struct store."""

import logging
import re
from abc import abstractmethod
from typing import Any, Callable, Dict, Generic, Optional, Sequence, TypeVar

from gpt_index.data_structs.table import BaseStructTable, StructDatapoint
from gpt_index.indices.base import DOCUMENTS_INPUT, BaseGPTIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TextSplitter
from gpt_index.prompts.default_prompts import DEFAULT_SCHEMA_EXTRACT_PROMPT
from gpt_index.prompts.prompts import SchemaExtractPrompt
from gpt_index.schema import BaseDocument
from gpt_index.utils import truncate_text

BST = TypeVar("BST", bound=BaseStructTable)


def default_output_parser(output: str) -> Optional[Dict[str, Any]]:
    """Parse output of schema extraction.

    Attempt to parse the following format from the default prompt:
    field1: <value>, field2: <value>, ...

    """
    tups = output.split("\n")

    fields = {}
    for tup in tups:
        if ":" in tup:
            tokens = tup.split(":")
            field = re.sub(r"\W+", "", tokens[0])
            value = re.sub(r"\W+", "", tokens[1])
            fields[field] = value
    return fields


OUTPUT_PARSER_TYPE = Callable[[str], Optional[Dict[str, Any]]]


class BaseGPTStructStoreIndex(BaseGPTIndex[BST], Generic[BST]):
    """Base GPT Struct Store Index."""

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[BST] = None,
        schema_extract_prompt: Optional[SchemaExtractPrompt] = None,
        output_parser: Optional[OUTPUT_PARSER_TYPE] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        text_splitter: Optional[TextSplitter] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self.schema_extract_prompt = (
            schema_extract_prompt or DEFAULT_SCHEMA_EXTRACT_PROMPT
        )
        self.output_parser = output_parser or default_output_parser
        super().__init__(
            documents=documents,
            index_struct=index_struct,
            llm_predictor=llm_predictor,
            text_splitter=text_splitter,
            **kwargs,
        )

    @abstractmethod
    def _insert_datapoint(self, datapoint: StructDatapoint) -> None:
        """Insert datapoint into index."""

    @abstractmethod
    def _get_col_types_map(self) -> Dict[str, type]:
        """Get col types map for schema."""

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
    def _get_schema_text(self) -> str:
        """Get schema text for extracting relevant info from unstructured text."""

    def _build_fallback_text_splitter(self) -> TextSplitter:
        # if not specified, use "smart" text splitter to ensure chunks fit in prompt
        return self._prompt_helper.get_text_splitter_given_prompt(
            self.schema_extract_prompt, 1
        )

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

    def _build_index_from_documents(self, documents: Sequence[BaseDocument]) -> BST:
        """Build index from documents."""
        index_struct = self.index_struct_cls()
        for d in documents:
            self._add_document_to_index(d)
        return index_struct

    def _insert(self, document: BaseDocument, **insert_kwargs: Any) -> None:
        """Insert a document."""
        self._add_document_to_index(document)

    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        raise NotImplementedError("Delete not implemented for Struct Store Index.")
