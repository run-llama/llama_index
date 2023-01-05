"""Struct store."""

from abc import abstractmethod
from typing import Any, Generic, Optional, Sequence, TypeVar, Callable, List

from gpt_index.data_structs.data_structs import BaseIndexDict
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.base import DOCUMENTS_INPUT, BaseGPTIndex
from gpt_index.indices.struct_store.schema import StructDatapoint, StructValue
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.default_prompts import DEFAULT_SCHEMA_EXTRACT_PROMPT
from gpt_index.prompts.prompts import SchemaExtractPrompt
from gpt_index.schema import BaseDocument
from gpt_index.indices.utils import truncate_text

BID = TypeVar("BID", bound=BaseIndexDict)


def default_output_parser(output: str) -> StructDatapoint:
    """Default output parser.
    
    Attempt to parse the following format from the default prompt:
    field1: <value>, field2: <value>, ...

    """
    tokens = output.split(",")
    
    for token in tokens:
        if ":" in token:
            return token.split(":")[1].strip()


class BaseGPTStructStoreIndex(BaseGPTIndex[BID], Generic[BID]):
    """Base GPT Struct Store Index."""

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[BID] = None,
        schema_extract_prompt: Optional[SchemaExtractPrompt] = None,
        schema_text: Optional[str] = None,
        output_parser: Optional[Callable[[str], str]] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if schema_text is None:
            raise ValueError("schema_text must be specified")
        self.schema_text = schema_text
        self.schema_extract_prompt = schema_extract_prompt or DEFAULT_SCHEMA_EXTRACT_PROMPT
        self.output_parser = output_parser or default_output_parser
        super().__init__(
            documents=documents,
            index_struct=index_struct,
            llm_predictor=llm_predictor,
            **kwargs,
        )

    def _add_document_to_index(
        self,
        index_struct: BID,
        document: BaseDocument,
        text_splitter: TokenTextSplitter,
    ) -> None:
        """Add document to index."""
        text_chunks = text_splitter.split_text(document.get_text())
        data = []
        for i, text_chunk in enumerate(text_chunks):
            fmt_text_chunk = truncate_text(text_chunk, 50)
            print(f"> Adding chunk: {fmt_text_chunk}")
            # if embedding specified in document, pass it to the Node
            response_str, _ = self._llm_predictor.predict(
                self.schema_extract_prompt,
                text=text_chunk,
            )
            struct_datapoint = self.output_parser(response_str)
            data.append(struct_datapoint)

        return nodes

    def _build_index_from_documents(
        self, documents: Sequence[BaseDocument], verbose: bool = False
    ) -> BID:
        """Build index from documents."""
        text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
            self.schema_extract_prompt, 1
        )
        index_struct = self.index_struct_cls()
        for d in documents:
            self._add_document_to_index(index_struct, d, text_splitter)
        return index_struct

    def _insert(self, document: BaseDocument, **insert_kwargs: Any) -> None:
        """Insert a document."""
        self._add_document_to_index(self._index_struct, document, self._text_splitter)