"""Top-level imports for LlamaIndex."""

import importlib
import logging
from importlib.metadata import PackageNotFoundError, version
from logging import NullHandler
from typing import Any, Callable, Optional

try:
    __version__ = version("llama-index-core")
except PackageNotFoundError:
    # This might happen when running tests or scripts directly without
    # an editable install.
    __version__ = "0.0.0"


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
    ListIndex,
    PropertyGraphIndex,
    RAKEKeywordTableIndex,
    SimpleKeywordTableIndex,
    SummaryIndex,
    TreeIndex,
    VectorStoreIndex,
    load_graph_from_storage,
    load_index_from_storage,
    load_indices_from_storage,
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
from llama_index.core.readers import SimpleDirectoryReader

# Response Synthesizer
from llama_index.core.response_synthesizers.factory import get_response_synthesizer
from llama_index.core.schema import Document, QueryBundle
from llama_index.core.service_context import (
    ServiceContext,
    set_global_service_context,
)

# global settings
from llama_index.core.settings import Settings

# storage
from llama_index.core.storage.storage_context import StorageContext

# global tokenizer
from llama_index.core.utils import get_tokenizer, set_global_tokenizer

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
    "Response",
    "Document",
    "SimpleDirectoryReader",
    "MockEmbedding",
    "SQLDatabase",
    "SQLDocumentContextBuilder",
    "SQLContextBuilder",
    "PromptHelper",
    "IndexStructType",
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

# global tokenizer
global_tokenizer: Optional[Callable[[str], list]] = None


# SQL support imports SQLAlchemy, which is slow to load, so these names are
# resolved on first access (PEP 562) instead of at package import time.
_LAZY_IMPORTS = {
    "SQLDatabase": "llama_index.core.utilities.sql_wrapper",
    "SQLDocumentContextBuilder": "llama_index.core.indices.common.struct_store.base",
}


def __getattr__(name: str) -> Any:
    if name == "SQLContextBuilder":  # NOTE: kept for backwards compatibility
        return __getattr__("SQLDocumentContextBuilder")
    if name in _LAZY_IMPORTS:
        value = getattr(importlib.import_module(_LAZY_IMPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
