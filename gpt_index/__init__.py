"""Init file of GPT Index."""

from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()


# embeddings
from gpt_index.embeddings.langchain import LangchainEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding

# indices
from gpt_index.indices.keyword_table import (
    GPTKeywordTableIndex,
    GPTRAKEKeywordTableIndex,
    GPTSimpleKeywordTableIndex,
)
from gpt_index.indices.list import GPTListIndex
from gpt_index.indices.tree import GPTTreeIndex
from gpt_index.indices.vector_store import (
    GPTFaissIndex,
    GPTSimpleVectorIndex,
    GPTWeaviateIndex,
)

# langchain helper
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor

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
    DiscordReader,
    Document,
    FaissReader,
    GoogleDocsReader,
    NotionPageReader,
    PineconeReader,
    SimpleDirectoryReader,
    SimpleMongoReader,
    SlackReader,
    WeaviateReader,
    WikipediaReader,
)

# token predictor
from gpt_index.token_predictor.mock_chain_wrapper import MockLLMPredictor

__all__ = [
    "GPTKeywordTableIndex",
    "GPTSimpleKeywordTableIndex",
    "GPTRAKEKeywordTableIndex",
    "GPTListIndex",
    "GPTTreeIndex",
    "GPTFaissIndex",
    "GPTSimpleVectorIndex",
    "GPTWeaviateIndex",
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
    "Document",
    "SimpleDirectoryReader",
    "SimpleMongoReader",
    "NotionPageReader",
    "GoogleDocsReader",
    "SlackReader",
    "WeaviateReader",
    "FaissReader",
    "PineconeReader",
    "DiscordReader",
    "LLMPredictor",
    "MockLLMPredictor",
]
