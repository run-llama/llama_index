"""Token splitter."""
from typing import Callable, List, Optional

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.text_splitter.types import TextSplitter
from llama_index.utils import globals_helper


class TokenTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at word tokens."""

    def __init__(
        self,
        separator: str = " ",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        tokenizer: Optional[Callable] = None,
        backup_separators: Optional[List[str]] = ["\n"],
        callback_manager: Optional[CallbackManager] = None,
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
        self._backup_separators = backup_separators
        self.callback_manager = callback_manager or CallbackManager([])

    def _reduce_chunk_size(
        self, start_idx: int, cur_idx: int, splits: List[str]
    ) -> int:
        """Reduce the chunk size by reducing cur_idx.

        Return the new cur_idx.

        """
        current_doc_total = len(
            self.tokenizer(self._separator.join(splits[start_idx:cur_idx]))
        )
        while current_doc_total > self._chunk_size:
            percent_to_reduce = (
                current_doc_total - self._chunk_size
            ) / current_doc_total
            num_to_reduce = int(percent_to_reduce * (cur_idx - start_idx)) + 1
            cur_idx -= num_to_reduce
            current_doc_total = len(
                self.tokenizer(self._separator.join(splits[start_idx:cur_idx]))
            )
        return cur_idx

    def _preprocess_splits(self, splits: List[str], chunk_size: int) -> List[str]:
        """Process splits.

        Specifically search for tokens that are too large for chunk size,
        and see if we can separate those tokens more
        (via backup separators if specified, or force chunking).

        """
        new_splits = []
        for split in splits:
            num_cur_tokens = len(self.tokenizer(split))
            if num_cur_tokens <= chunk_size:
                new_splits.append(split)
            else:
                cur_splits = [split]
                if self._backup_separators:
                    for sep in self._backup_separators:
                        if sep in split:
                            cur_splits = split.split(sep)
                            break
                else:
                    cur_splits = [split]

                cur_splits2 = []
                for cur_split in cur_splits:
                    num_cur_tokens = len(self.tokenizer(cur_split))
                    if num_cur_tokens <= chunk_size:
                        cur_splits2.extend([cur_split])
                    else:
                        # split cur_split according to chunk size of the token numbers
                        cur_split_chunks = []
                        end_idx = len(cur_split)
                        while len(self.tokenizer(cur_split[0:end_idx])) > chunk_size:
                            for i in range(1, end_idx):
                                tmp_split = cur_split[0 : end_idx - i]
                                if len(self.tokenizer(tmp_split)) <= chunk_size:
                                    cur_split_chunks.append(tmp_split)
                                    cur_split = cur_split[end_idx - i : end_idx]
                                    end_idx = len(cur_split)
                                    break
                        cur_split_chunks.append(cur_split)
                        cur_splits2.extend(cur_split_chunks)

                new_splits.extend(cur_splits2)
        return new_splits

    def _postprocess_splits(self, docs: List[str]) -> List[str]:
        """Post-process splits."""
        # TODO: prune text splits, remove empty spaces
        new_docs = []
        for doc in docs:
            if doc.replace(" ", "") == "":
                continue
            new_docs.append(doc)
        return new_docs

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ) as event:
            chunks = self._split_text(text)

            event.on_end(
                payload={EventPayload.CHUNKS: chunks},
            )

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks with overlap size."""
        if text == "":
            return []

        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ) as event:
            # First we naively split the large input into a bunch of smaller ones.
            splits = text.split(self._separator)
            splits = self._preprocess_splits(splits, self._chunk_size)
            # We now want to combine these smaller pieces into medium size
            # chunks to send to the LLM.
            docs: List[str] = []

            start_idx = 0
            cur_idx = 0
            cur_total = 0
            prev_idx = 0  # store the previous end index
            while cur_idx < len(splits):
                cur_token = splits[cur_idx]
                num_cur_tokens = max(len(self.tokenizer(cur_token)), 1)
                if num_cur_tokens > self._chunk_size:
                    raise ValueError(
                        "A single term is larger than the allowed chunk size.\n"
                        f"Term size: {num_cur_tokens}\n"
                        f"Chunk size: {self._chunk_size}"
                        f"Effective chunk size: {self._chunk_size}"
                    )
                # If adding token to current_doc would exceed the chunk size:
                # 1. First verify with tokenizer that current_doc
                # 1. Update the docs list
                if cur_total + num_cur_tokens > self._chunk_size:
                    # NOTE: since we use a proxy for counting tokens, we want to
                    # run tokenizer across all of current_doc first. If
                    # the chunk is too big, then we will reduce text in pieces
                    cur_idx = self._reduce_chunk_size(start_idx, cur_idx, splits)
                    overlap = 0
                    # after first round, check if last chunk
                    # ended after this chunk begins
                    if prev_idx > 0 and prev_idx > start_idx:
                        overlap = sum(
                            [len(splits[i]) for i in range(start_idx, prev_idx)]
                        )

                    docs.append(self._separator.join(splits[start_idx:cur_idx]))
                    prev_idx = cur_idx
                    # 2. Shrink the current_doc (from the front) until it is gets
                    # smaller than the overlap size
                    # NOTE: because counting tokens individually is an imperfect
                    # proxy (but much faster proxy) for the total number of tokens
                    # consumed, we need to enforce that start_idx <= cur_idx, otherwise
                    # start_idx has a chance of going out of bounds.
                    while cur_total > self._chunk_overlap and start_idx < cur_idx:
                        # # call tokenizer on entire overlap
                        # cur_total = self.tokenizer()
                        cur_num_tokens = max(len(self.tokenizer(splits[start_idx])), 1)
                        cur_total -= cur_num_tokens
                        start_idx += 1
                    # NOTE: This is a hack, make more general
                    if start_idx == cur_idx:
                        cur_total = 0
                # Build up the current_doc with term d, and update the total counter
                # with the number of the number of tokens in d, wrt self.tokenizer

                # we reassign cur_token and num_cur_tokens, because cur_idx
                # may have changed
                cur_token = splits[cur_idx]
                num_cur_tokens = max(len(self.tokenizer(cur_token)), 1)

                cur_total += num_cur_tokens
                cur_idx += 1
            overlap = 0
            # after first round, check if last chunk ended after this chunk begins
            if prev_idx > start_idx:
                overlap = sum(
                    [len(splits[i]) for i in range(start_idx, prev_idx)]
                ) + len(range(start_idx, prev_idx))
            docs.append(self._separator.join(splits[start_idx:cur_idx]))

            # run postprocessing to remove blank spaces
            docs = self._postprocess_splits(docs)

            event.on_end(payload={EventPayload.CHUNKS: docs})

        return docs
