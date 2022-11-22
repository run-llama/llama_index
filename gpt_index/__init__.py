"""Init file of GPT Index."""

from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()


from gpt_index.indices.keyword_table.base import GPTKeywordTableIndex
from gpt_index.indices.list.base import GPTListIndex

# indices
from gpt_index.indices.tree.base import GPTTreeIndex

# prompts
from gpt_index.prompts.base import Prompt

# readers
from gpt_index.readers.simple_reader import SimpleDirectoryReader
from gpt_index.readers.wikipedia import WikipediaReader

__all__ = [
    "GPTKeywordTableIndex",
    "GPTListIndex",
    "GPTTreeIndex",
    "Prompt",
    "WikipediaReader",
    "SimpleDirectoryReader",
]
