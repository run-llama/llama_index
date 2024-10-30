"""Init file of LlamaIndex."""

__version__ = "0.11.20"

import logging
from logging import NullHandler
from typing import Callable, Optional

# response
from llama_index.core.base.response.schema import Response

# import global eval handler
from llama_index.core.callbacks.global_handlers import set_global_handler
from llama_index.core.data_structs.struct_type import IndexStructType
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

# indices
# loading
from llama_index.core.indices import (
    ComposableGraph,
    DocumentSummaryIndex,
    GPTDocumentSummaryIndex,
    GPTKeywordTableIndex,
    GPTListIndex,
    GPTRAKEKeywordTableIndex,
    GPTSimpleKeywordTableIndex,
    GPTTreeIndex,
    GPTVectorStoreIndex,
    KeywordTableIndex,
    KnowledgeGraphIndex,
    PropertyGraphIndex,
    ListIndex,
    RAKEKeywordTableIndex,
    SimpleKeywordTableIndex,
    SummaryIndex,
    TreeIndex,
    VectorStoreIndex,
    load_graph_from_storage,
    load_index_from_storage,
    load_indices_from_storage,
)

# structured
from llama_index.core.indices.common.struct_store.base import (
    SQLDocumentContextBuilder,
)

# prompt helper
from llama_index.core.indices.prompt_helper import PromptHelper

# prompts
from llama_index.core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    # backwards compatibility
    Prompt,
    PromptTemplate,
    SelectorPromptTemplate,
)
from llama_index.core.readers import SimpleDirectoryReader, download_loader

# Response Synthesizer
from llama_index.core.response_synthesizers.factory import get_response_synthesizer
from llama_index.core.schema import Document, QueryBundle
from llama_index.core.service_context import (
    ServiceContext,
    set_global_service_context,
)

# storage
from llama_index.core.storage.storage_context import StorageContext

# sql wrapper
from llama_index.core.utilities.sql_wrapper import SQLDatabase

# global tokenizer
from llama_index.core.utils import get_tokenizer, set_global_tokenizer

# global settings
from llama_index.core.settings import Settings

# best practices for library logging:
# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logging.getLogger(__name__).addHandler(NullHandler())

__all__ = [
    "StorageContext",
    "ServiceContext",
    "ComposableGraph",
    # indices
    "SummaryIndex",
    "VectorStoreIndex",
    "SimpleKeywordTableIndex",
    "KeywordTableIndex",
    "RAKEKeywordTableIndex",
    "TreeIndex",
    "DocumentSummaryIndex",
    "KnowledgeGraphIndex",
    "PropertyGraphIndex",
    # indices - legacy names
    "GPTKeywordTableIndex",
    "GPTKnowledgeGraphIndex",
    "GPTSimpleKeywordTableIndex",
    "GPTRAKEKeywordTableIndex",
    "GPTListIndex",
    "ListIndex",
    "GPTTreeIndex",
    "GPTVectorStoreIndex",
    "GPTDocumentSummaryIndex",
    "Prompt",
    "PromptTemplate",
    "BasePromptTemplate",
    "ChatPromptTemplate",
    "SelectorPromptTemplate",
    "SummaryPrompt",
    "TreeInsertPrompt",
    "TreeSelectPrompt",
    "TreeSelectMultiplePrompt",
    "RefinePrompt",
    "QuestionAnswerPrompt",
    "KeywordExtractPrompt",
    "QueryKeywordExtractPrompt",
    "Response",
    "Document",
    "SimpleDirectoryReader",
    "VellumPredictor",
    "VellumPromptRegistry",
    "MockEmbedding",
    "SQLDatabase",
    "SQLDocumentContextBuilder",
    "SQLContextBuilder",
    "PromptHelper",
    "IndexStructType",
    "download_loader",
    "load_graph_from_storage",
    "load_index_from_storage",
    "load_indices_from_storage",
    "QueryBundle",
    "get_response_synthesizer",
    "set_global_service_context",
    "set_global_handler",
    "set_global_tokenizer",
    "get_tokenizer",
    "Settings",
]

# eval global toggle
from llama_index.core.callbacks.base_handler import BaseCallbackHandler

global_handler: Optional[BaseCallbackHandler] = None

# NOTE: keep for backwards compatibility
SQLContextBuilder = SQLDocumentContextBuilder

# global tokenizer
global_tokenizer: Optional[Callable[[str], list]] = None
