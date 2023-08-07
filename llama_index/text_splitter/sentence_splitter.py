"""Sentence splitter."""
from dataclasses import dataclass
from typing import Callable, List, Optional

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.constants import DEFAULT_CHUNK_SIZE
from llama_index.text_splitter.types import TextSplitter
from llama_index.utils import globals_helper


class SentenceSplitter(TextSplitter):
    """Split text with a preference for complete sentences.

    In general, this class tries to keep sentences and paragraphs together. Therefore
    compared to the original TokenTextSplitter, there are less likely to be
    hanging sentences or parts of sentences at the end of the node chunk.
    """

    def __init__(
        self,
        separator: str = " ",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = 200,
        tokenizer: Optional[Callable] = None,
        backup_separators: Optional[List[str]] = ["\n"],
        paragraph_separator: Optional[str] = "\n\n\n",
        chunking_tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
        secondary_chunking_regex: Optional[str] = "[^,.;。]+[,.;。]?",
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
        if chunking_tokenizer_fn is None:
            import nltk.tokenize.punkt as pkt

            class CustomLanguageVars(pkt.PunktLanguageVars):
                _period_context_fmt = r"""
                    %(SentEndChars)s             # a potential sentence ending
                    (\)\"\s)\s*                  # other end chars and
                                                 # any amount of white space
                    (?=(?P<after_tok>
                        %(NonWord)s              # either other punctuation
                        |
                        (?P<next_tok>\S+)     # or whitespace and some other token
                    ))"""

            custom_tknzr = pkt.PunktSentenceTokenizer(lang_vars=CustomLanguageVars())

            chunking_tokenizer_fn = custom_tknzr.tokenize
        self.paragraph_separator = paragraph_separator
        self.chunking_tokenizer_fn = chunking_tokenizer_fn
        self.second_chunking_regex = secondary_chunking_regex
        """
        By default we use the second chunking regex "[^,.;]+[,.;]?".
        This regular expression will split the sentences into phrases,
        where each phrase is a sequence of one or more non-comma,
        non-period, and non-semicolon characters, followed by an optional comma,
        period, or semicolon. The regular expression will also capture the
        delimiters themselves as separate items in the list of phrases.
        """

    def _postprocess_splits(self, docs: List[str]) -> List[str]:
        """Post-process splits."""
        # TODO: prune text splits, remove empty spaces
        new_docs = []
        for doc in docs:
            if doc.replace(" ", "") == "":
                continue
            new_docs.append(doc)
        return new_docs

    def _split_text(self, text: str) -> List[str]:
        """
        Split incoming text and return chunks with overlap size.

        Has a preference for complete sentences, phrases, and minimal overlap.
        """
        if text == "":
            return []

        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ) as event:
            # First we split paragraphs using separator
            splits = text.split(self.paragraph_separator)

            # Merge paragraphs that are too small.

            idx = 0
            while idx < len(splits):
                if idx < len(splits) - 1 and len(splits[idx]) < self._chunk_size:
                    splits[idx] = "\n\n".join([splits[idx], splits[idx + 1]])
                    splits.pop(idx + 1)
                else:
                    idx += 1

            # Next we split the text using the chunk tokenizer fn,
            # which defaults to the sentence tokenizer from nltk.
            chunked_splits = [self.chunking_tokenizer_fn(text) for text in splits]
            splits = [chunk for split in chunked_splits for chunk in split]

            # Check if any sentences exceed the chunk size. If they do, split again
            # using the second chunk separator. If it any still exceed,
            # use the default separator (" ").
            @dataclass
            class Split:
                text: str  # the split text
                is_sentence: bool  # save whether this is a full sentence

            new_splits: List[Split] = []
            for split in splits:
                split_len = len(self.tokenizer(split))
                if split_len <= self._chunk_size:
                    new_splits.append(Split(split, True))
                else:
                    if self.second_chunking_regex is not None:
                        import re

                        # Default regex is "[^,\.;]+[,\.;]?"
                        splits2 = re.findall(self.second_chunking_regex, split)

                    else:
                        splits2 = [split]
                    for split2 in splits2:
                        if len(self.tokenizer(split2)) <= self._chunk_size:
                            new_splits.append(Split(split2, False))
                        else:
                            splits3 = split2.split(self._separator)
                            new_splits.extend(
                                [Split(split3, False) for split3 in splits3]
                            )

            # Create the list of text splits by combining smaller chunks.
            docs: List[str] = []
            cur_doc_list: List[str] = []
            cur_tokens = 0
            while len(new_splits) > 0:
                cur_token = new_splits[0]
                cur_len = len(self.tokenizer(cur_token.text))
                if cur_len > self._chunk_size:
                    raise ValueError("Single token exceed chunk size")
                if cur_tokens + cur_len > self._chunk_size:
                    docs.append("".join(cur_doc_list).strip())
                    cur_doc_list = []
                    cur_tokens = 0
                else:
                    if (
                        cur_token.is_sentence
                        or cur_tokens + cur_len
                        < self._chunk_size - self._chunk_overlap
                    ):
                        cur_tokens += cur_len
                        cur_doc_list.append(cur_token.text)
                        new_splits.pop(0)
                    else:
                        docs.append("".join(cur_doc_list).strip())
                        cur_doc_list = []
                        cur_tokens = 0

            docs.append("".join(cur_doc_list).strip())

            # run postprocessing to remove blank spaces
            docs = self._postprocess_splits(docs)

            event.on_end(payload={EventPayload.CHUNKS: [x for x in docs]})

        return docs

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ) as event:
            text_splits = self.split_text_with_overlaps(text)
            chunks = [text_split.text_chunk for text_split in text_splits]

            event.on_end(
                payload={EventPayload.CHUNKS: chunks},
            )

        return chunks
