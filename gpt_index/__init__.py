"""Init file of GPT Index."""

from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()

from gpt_index.indices.keyword_table.base import GPTKeywordTableIndex

# indices
from gpt_index.indices.tree.base import GPTTreeIndex

# readers
from gpt_index.readers.simple_reader import SimpleDirectoryReader

__all__ = ["GPTTreeIndex", "SimpleDirectoryReader", "GPTKeywordTableIndex"]
