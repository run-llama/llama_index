"""Struct store."""

import re
from typing import Any, Callable, Dict, Generic, Optional, Sequence, TypeVar

from llama_index.data_structs.table import BaseStructTable
from llama_index.indices.base import BaseIndex
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts import BasePromptTemplate
from llama_index.prompts.default_prompts import DEFAULT_SCHEMA_EXTRACT_PROMPT
from llama_index.schema import BaseNode
from llama_index.storage.docstore.types import RefDocInfo

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


class BaseStructStoreIndex(BaseIndex[BST], Generic[BST]):
    """Base Struct Store Index."""

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[BST] = None,
        service_context: Optional[ServiceContext] = None,
        schema_extract_prompt: Optional[BasePromptTemplate] = None,
        output_parser: Optional[OUTPUT_PARSER_TYPE] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self.schema_extract_prompt = (
            schema_extract_prompt or DEFAULT_SCHEMA_EXTRACT_PROMPT
        )
        self.output_parser = output_parser or default_output_parser
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            **kwargs,
        )

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        raise NotImplementedError("Delete not implemented for Struct Store Index.")

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        raise NotImplementedError("Struct Store Index does not support ref_doc_info.")
