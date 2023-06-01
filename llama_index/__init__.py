"""Init file of LlamaIndex."""
from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()


import logging
from logging import NullHandler
from typing import Optional

from llama_index.data_structs.struct_type import IndexStructType

# embeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# structured
from llama_index.indices.common.struct_store.base import SQLDocumentContextBuilder
from llama_index.indices.composability.graph import ComposableGraph
from llama_index.indices.empty import GPTEmptyIndex

# indices
from llama_index.indices.keyword_table import (
    GPTKeywordTableIndex,
    GPTRAKEKeywordTableIndex,
    GPTSimpleKeywordTableIndex,
)
from llama_index.indices.list import GPTListIndex

# loading
from llama_index.indices.loading import (
    load_graph_from_storage,
    load_index_from_storage,
    load_indices_from_storage,
)

# prompt helper
from llama_index.indices.prompt_helper import PromptHelper

# Response Synthesizer
from llama_index.indices.query.response_synthesis import ResponseSynthesizer

# QueryBundle
from llama_index.indices.query.schema import QueryBundle

# for composability
from llama_index.indices.service_context import (
    ServiceContext,
    set_global_service_context,
)
from llama_index.indices.struct_store.sql import GPTSQLStructStoreIndex
from llama_index.indices.tree import GPTTreeIndex
from llama_index.indices.vector_store import GPTVectorStoreIndex

# langchain helper
from llama_index.langchain_helpers.chain_wrapper import LLMPredictor
from llama_index.langchain_helpers.memory_wrapper import GPTIndexMemory
from llama_index.langchain_helpers.sql_wrapper import SQLDatabase

# prompts
from llama_index.prompts.base import Prompt
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

# readers
from llama_index.readers import (
    BeautifulSoupWebReader,
    ChromaReader,
    DeepLakeReader,
    DiscordReader,
    Document,
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

# storage
from llama_index.storage.storage_context import StorageContext

# token predictor
from llama_index.token_counter.mock_chain_wrapper import MockLLMPredictor
from llama_index.token_counter.mock_embed_model import MockEmbedding

# vellum
from llama_index.llm_predictor.vellum import VellumPredictor, VellumPromptRegistry

# best practices for library logging:
# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logging.getLogger(__name__).addHandler(NullHandler())


__all__ = [
    "StorageContext",
    "ServiceContext",
    "ComposableGraph",
    "GPTKeywordTableIndex",
    "GPTSimpleKeywordTableIndex",
    "GPTRAKEKeywordTableIndex",
    "GPTListIndex",
    "GPTEmptyIndex",
    "GPTTreeIndex",
    "GPTVectorStoreIndex",
    "GPTSQLStructStoreIndex",
    "Prompt",
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
    "ResponseSynthesizer",
    "set_global_service_context",
]

# NOTE: keep for backwards compatibility
SQLContextBuilder = SQLDocumentContextBuilder

# global service context for ServiceContext.from_defaults()
global_service_context: Optional[ServiceContext] = None
