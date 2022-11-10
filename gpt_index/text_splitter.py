"""Text splitter implementations."""
from typing import List

from langchain.text_splitter import TextSplitter
from transformers import GPT2TokenizerFast


class TokenTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at word tokens."""

    def __init__(
        self, separator: str = " ", chunk_size: int = 4000, chunk_overlap: int = 200
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
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        splits = text.split(self._separator)
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            if total > self._chunk_size:
                docs.append(self._separator.join(current_doc))
                while total > self._chunk_overlap:
                    cur_tokens = self.tokenizer(current_doc[0])
                    total -= len(cur_tokens["input_ids"])
                    current_doc = current_doc[1:]
            current_doc.append(d)
            num_tokens = len(self.tokenizer(d)["input_ids"])
            total += num_tokens
        docs.append(self._separator.join(current_doc))
        return docs
