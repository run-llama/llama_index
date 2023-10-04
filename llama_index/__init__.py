"""Init file of LlamaIndex."""
from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()


import logging
from logging import NullHandler
from typing import Optional

# import global eval handler
from llama_index.callbacks.global_handlers import set_global_handler
from llama_index.data_structs.struct_type import IndexStructType

# embeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# structured
from llama_index.indices.common.struct_store.base import SQLDocumentContextBuilder

# for composability
from llama_index.indices.composability.graph import ComposableGraph
from llama_index.indices.document_summary import (
    DocumentSummaryIndex,
    GPTDocumentSummaryIndex,
)
from llama_index.indices.empty import EmptyIndex, GPTEmptyIndex

# indices
from llama_index.indices.keyword_table import (
    GPTKeywordTableIndex,
    GPTRAKEKeywordTableIndex,
    GPTSimpleKeywordTableIndex,
    KeywordTableIndex,
    RAKEKeywordTableIndex,
    SimpleKeywordTableIndex,
)
from llama_index.indices.knowledge_graph import (
    GPTKnowledgeGraphIndex,
    KnowledgeGraphIndex,
)
from llama_index.indices.list import GPTListIndex, ListIndex, SummaryIndex

# loading
from llama_index.indices.loading import (
    load_graph_from_storage,
    load_index_from_storage,
    load_indices_from_storage,
)

# prompt helper
from llama_index.indices.prompt_helper import PromptHelper

# QueryBundle
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import (
    ServiceContext,
    set_global_service_context,
)
from llama_index.indices.struct_store.pandas import GPTPandasIndex, PandasIndex
from llama_index.indices.struct_store.sql import (
    GPTSQLStructStoreIndex,
    SQLStructStoreIndex,
)
from llama_index.indices.tree import GPTTreeIndex, TreeIndex
from llama_index.indices.vector_store import GPTVectorStoreIndex, VectorStoreIndex
from llama_index.langchain_helpers.memory_wrapper import GPTIndexMemory

# langchain helper
from llama_index.llm_predictor import LLMPredictor

# token predictor
from llama_index.llm_predictor.mock import MockLLMPredictor

# vellum
from llama_index.llm_predictor.vellum import VellumPredictor, VellumPromptRegistry

# prompts
from llama_index.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    # backwards compatibility
    Prompt,
    PromptTemplate,
    SelectorPromptTemplate,
)
from llama_index.prompts.prompts import (
    KeywordExtractPrompt,
    QueryKeywordExtractPrompt,
    QuestionAnswerPrompt,
    RefinePrompt,
    SummaryPrompt,
    TreeInsertPrompt,
    TreeSelectMultiplePrompt,
    TreeSelectPrompt,
)
from llama_index.readers import (
    BeautifulSoupWebReader,
    ChromaReader,
    DeepLakeReader,
    DiscordReader,
    FaissReader,
    GithubRepositoryReader,
    GoogleDocsReader,
    JSONReader,
    MboxReader,
    MilvusReader,
    NotionPageReader,
    ObsidianReader,
    PineconeReader,
    PsychicReader,
    QdrantReader,
    RssReader,
    SimpleDirectoryReader,
    SimpleMongoReader,
    SimpleWebPageReader,
    SlackReader,
    StringIterableReader,
    TrafilaturaWebReader,
    TwitterTweetReader,
    WeaviateReader,
    WikipediaReader,
)
from llama_index.readers.download import download_loader

# response
from llama_index.response.schema import Response

# Response Synthesizer
from llama_index.response_synthesizers.factory import get_response_synthesizer

# readers
from llama_index.schema import Document

# storage
from llama_index.storage.storage_context import StorageContext
from llama_index.token_counter.mock_embed_model import MockEmbedding

# sql wrapper
from llama_index.utilities.sql_wrapper import SQLDatabase

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
    "SQLStructStoreIndex",
    "PandasIndex",
    "EmptyIndex",
    "DocumentSummaryIndex",
    "KnowledgeGraphIndex",
    # indices - legacy names
    "GPTKeywordTableIndex",
    "GPTKnowledgeGraphIndex",
    "GPTSimpleKeywordTableIndex",
    "GPTRAKEKeywordTableIndex",
    "GPTListIndex",
    "ListIndex",
    "GPTEmptyIndex",
    "GPTTreeIndex",
    "GPTVectorStoreIndex",
    "GPTPandasIndex",
    "GPTSQLStructStoreIndex",
    "GPTDocumentSummaryIndex",
    "Prompt",
    "PromptTemplate",
    "BasePromptTemplate",
    "ChatPromptTemplate",
    "SelectorPromptTemplate",
    "LangchainEmbedding",
    "OpenAIEmbedding",
    "SummaryPrompt",
    "TreeInsertPrompt",
    "TreeSelectPrompt",
    "TreeSelectMultiplePrompt",
    "RefinePrompt",
    "QuestionAnswerPrompt",
    "KeywordExtractPrompt",
    "QueryKeywordExtractPrompt",
    "Response",
    "WikipediaReader",
    "ObsidianReader",
    "Document",
    "SimpleDirectoryReader",
    "JSONReader",
    "SimpleMongoReader",
    "NotionPageReader",
    "GoogleDocsReader",
    "MboxReader",
    "SlackReader",
    "StringIterableReader",
    "WeaviateReader",
    "FaissReader",
    "ChromaReader",
    "DeepLakeReader",
    "PineconeReader",
    "PsychicReader",
    "QdrantReader",
    "MilvusReader",
    "DiscordReader",
    "SimpleWebPageReader",
    "RssReader",
    "BeautifulSoupWebReader",
    "TrafilaturaWebReader",
    "LLMPredictor",
    "MockLLMPredictor",
    "VellumPredictor",
    "VellumPromptRegistry",
    "MockEmbedding",
    "SQLDatabase",
    "GPTIndexMemory",
    "SQLDocumentContextBuilder",
    "SQLContextBuilder",
    "PromptHelper",
    "IndexStructType",
    "TwitterTweetReader",
    "download_loader",
    "GithubRepositoryReader",
    "load_graph_from_storage",
    "load_index_from_storage",
    "load_indices_from_storage",
    "QueryBundle",
    "get_response_synthesizer",
    "set_global_service_context",
    "set_global_handler",
]

# eval global toggle
from llama_index.callbacks.base_handler import BaseCallbackHandler

global_handler: Optional[BaseCallbackHandler] = None

# NOTE: keep for backwards compatibility
SQLContextBuilder = SQLDocumentContextBuilder

# global service context for ServiceContext.from_defaults()
global_service_context: Optional[ServiceContext] = None
