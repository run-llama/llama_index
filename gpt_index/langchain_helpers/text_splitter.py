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
            num_tokens = len(self.tokenizer(d))
            # If the total tokens in current_doc exceeds the chunk size:
            # 1. Update the docs list
            if total + num_tokens > self._chunk_size:
                docs.append(self._separator.join(current_doc))
                # 2. Shrink the current_doc (from the front) until it is gets smaller
                # than the overlap size
                while total > self._chunk_overlap:
                    cur_tokens = self.tokenizer(current_doc[0])
                    total -= len(cur_tokens)
                    current_doc = current_doc[1:]
                # 3. From here we can continue to build up the current_doc again
            # Build up the current_doc with term d, and update the total counter with
            # the number of the number of tokens in d, wrt self.tokenizer
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
            num_tokens = len(self.tokenizer(d))
            if total + num_tokens > self._chunk_size:
                break
            current_doc.append(d)
            total += num_tokens
        return self._separator.join(current_doc)
