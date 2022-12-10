"""Text splitter implementations."""
from typing import Callable, List, Optional

from langchain.text_splitter import TextSplitter

from gpt_index.utils import globals_helper


class TokenTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at word tokens."""

    def __init__(
        self,
        separator: str = " ",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        tokenizer: Optional[Callable] = None,
    ):
        """Initialize with parameters."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._separator = separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer or globals_helper.tokenizer

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        if text == "":
            return []
        # First we naively split the large input into a bunch of smaller ones.
        splits = text.split(self._separator)
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            num_tokens = len(self.tokenizer(d)["input_ids"])
            if total + num_tokens > self._chunk_size:
                docs.append(self._separator.join(current_doc))
                while total > self._chunk_overlap:
                    cur_tokens = self.tokenizer(current_doc[0])
                    total -= len(cur_tokens["input_ids"])
                    current_doc = current_doc[1:]
            current_doc.append(d)
            total += num_tokens
        docs.append(self._separator.join(current_doc))
        return docs

    def truncate_text(self, text: str) -> str:
        """Truncate text in order to fit the underlying chunk size."""
        if text == "":
            return ""
        # First we naively split the large input into a bunch of smaller ones.
        splits = text.split(self._separator)
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        current_doc: List[str] = []
        total = 0
        for d in splits:
            num_tokens = len(self.tokenizer(d)["input_ids"])
            if total + num_tokens > self._chunk_size:
                break
            current_doc.append(d)
            total += num_tokens
        return self._separator.join(current_doc)
