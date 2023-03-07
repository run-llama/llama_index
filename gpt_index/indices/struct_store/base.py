"""Struct store."""

import re
from typing import Any, Callable, Dict, Generic, Optional, Sequence, TypeVar

from gpt_index.data_structs.table import BaseStructTable
from gpt_index.indices.base import DOCUMENTS_INPUT, BaseGPTIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TextSplitter
from gpt_index.prompts.default_prompts import DEFAULT_SCHEMA_EXTRACT_PROMPT
from gpt_index.prompts.prompts import SchemaExtractPrompt

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

    def _build_fallback_text_splitter(self) -> TextSplitter:
        # if not specified, use "smart" text splitter to ensure chunks fit in prompt
        return self._prompt_helper.get_text_splitter_given_prompt(
            self.schema_extract_prompt, 1
        )

    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        raise NotImplementedError("Delete not implemented for Struct Store Index.")
