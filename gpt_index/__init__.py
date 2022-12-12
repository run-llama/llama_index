"""Init file of GPT Index."""

from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()


# indices
from gpt_index.indices.keyword_table import (
    GPTKeywordTableIndex,
    GPTRAKEKeywordTableIndex,
    GPTSimpleKeywordTableIndex,
)
from gpt_index.indices.list import GPTListIndex
from gpt_index.indices.tree import GPTTreeIndex

# langchain helper
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor

# prompts
from gpt_index.prompts.base import Prompt

# readers
from gpt_index.readers.file import SimpleDirectoryReader
from gpt_index.readers.google.gdocs import GoogleDocsReader
from gpt_index.readers.mongo import SimpleMongoReader
from gpt_index.readers.notion import NotionPageReader
from gpt_index.readers.slack import SlackReader
from gpt_index.readers.wikipedia import WikipediaReader

__all__ = [
    "GPTKeywordTableIndex",
    "GPTSimpleKeywordTableIndex",
    "GPTRAKEKeywordTableIndex",
    "GPTListIndex",
    "GPTTreeIndex",
    "Prompt",
    "WikipediaReader",
    "SimpleDirectoryReader",
    "SimpleMongoReader",
    "NotionPageReader",
    "GoogleDocsReader",
    "SlackReader",
    "LLMPredictor",
]
