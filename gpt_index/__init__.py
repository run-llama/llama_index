"""Init file of LlamaIndex."""
from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()


import logging
from logging import NullHandler

from gpt_index.data_structs.struct_type import IndexStructType

# embeddings
from gpt_index.embeddings.langchain import LangchainEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding

# structured
from gpt_index.indices.common.struct_store.base import SQLDocumentContextBuilder

# indices
from gpt_index.indices.keyword_table import (
    GPTKeywordTableIndex,
    GPTRAKEKeywordTableIndex,
    GPTSimpleKeywordTableIndex,
)
from gpt_index.indices.list import GPTListIndex

# prompt helper
from gpt_index.indices.prompt_helper import PromptHelper

# for composability
from gpt_index.indices.query.schema import QueryConfig, QueryMode
from gpt_index.indices.struct_store.sql import GPTSQLStructStoreIndex
from gpt_index.indices.tree import GPTTreeIndex
from gpt_index.indices.vector_store import (
    GPTChromaIndex,
    GPTFaissIndex,
    GPTPineconeIndex,
    GPTQdrantIndex,
    GPTSimpleVectorIndex,
    GPTVectorStoreIndex,
    GPTWeaviateIndex,
)

# langchain helper
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.memory_wrapper import GPTIndexMemory
from gpt_index.langchain_helpers.sql_wrapper import SQLDatabase

# prompts
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.prompts import (
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
from gpt_index.readers import (
    BeautifulSoupWebReader,
    ChromaReader,
    DiscordReader,
    Document,
    FaissReader,
    GithubRepositoryReader,
    GoogleDocsReader,
    MboxReader,
    NotionPageReader,
    ObsidianReader,
    PineconeReader,
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
from gpt_index.readers.download import download_loader

# token predictor
from gpt_index.token_counter.mock_chain_wrapper import MockLLMPredictor
from gpt_index.token_counter.mock_embed_model import MockEmbedding

# best practices for library logging:
# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logging.getLogger(__name__).addHandler(NullHandler())


__all__ = [
    "GPTKeywordTableIndex",
    "GPTSimpleKeywordTableIndex",
    "GPTRAKEKeywordTableIndex",
    "GPTListIndex",
    "GPTTreeIndex",
    "GPTFaissIndex",
    "GPTPineconeIndex",
    "GPTQdrantIndex",
    "GPTSimpleVectorIndex",
    "GPTVectorStoreIndex",
    "GPTWeaviateIndex",
    "GPTChromaIndex",
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
    "WikipediaReader",
    "ObsidianReader",
    "Document",
    "SimpleDirectoryReader",
    "SimpleMongoReader",
    "NotionPageReader",
    "GoogleDocsReader",
    "MboxReader",
    "SlackReader",
    "StringIterableReader",
    "WeaviateReader",
    "FaissReader",
    "ChromaReader",
    "PineconeReader",
    "QdrantReader",
    "DiscordReader",
    "SimpleWebPageReader",
    "RssReader",
    "BeautifulSoupWebReader",
    "TrafilaturaWebReader",
    "LLMPredictor",
    "MockLLMPredictor",
    "MockEmbedding",
    "SQLDatabase",
    "GPTIndexMemory",
    "SQLDocumentContextBuilder",
    "SQLContextBuilder",
    "PromptHelper",
    "QueryConfig",
    "QueryMode",
    "IndexStructType",
    "TwitterTweetReader",
    "download_loader",
    "GithubRepositoryReader",
]

# NOTE: keep for backwards compatibility
SQLContextBuilder = SQLDocumentContextBuilder
