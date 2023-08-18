"""General prompt helper that can help deal with LLM context window token limitations.

At its core, it calculates available context size by starting with the context window 
size of an LLM and reserve token space for the prompt template, and the output.

It provides utility for "repacking" text chunks (retrieved from index) to maximally 
make use of the available context window (and thereby reducing the number of LLM calls 
needed), or truncating them so that they fit in a single LLM call.
"""

import logging
from typing import Callable, List, Optional, Sequence

from llama_index.prompts import BasePromptTemplate

from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.llm_predictor.base import LLMMetadata
from llama_index.prompts.prompt_utils import get_empty_prompt_txt
from llama_index.text_splitter import TokenTextSplitter
from llama_index.text_splitter.utils import truncate_text
from llama_index.utils import globals_helper

DEFAULT_PADDING = 5
DEFAULT_CHUNK_OVERLAP_RATIO = 0.1

logger = logging.getLogger(__name__)


class PromptHelper:
    """Prompt helper.

    General prompt helper that can help deal with LLM context window token limitations.

    At its core, it calculates available context size by starting with the context
    window size of an LLM and reserve token space for the prompt template, and the
    output.

    It provides utility for "repacking" text chunks (retrieved from index) to maximally
    make use of the available context window (and thereby reducing the number of LLM
    calls needed), or truncating them so that they fit in a single LLM call.

    Args:
        context_window (int):                   Context window for the LLM.
        num_output (int):                       Number of outputs for the LLM.
        chunk_overlap_ratio (float):            Chunk overlap as a ratio of chunk size
        chunk_size_limit (Optional[int]):         Maximum chunk size to use.
        tokenizer (Optional[Callable[[str], List]]): Tokenizer to use.
        separator (str):                        Separator for text splitter

    """

    def __init__(
        self,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        num_output: int = DEFAULT_NUM_OUTPUTS,
        chunk_overlap_ratio: float = DEFAULT_CHUNK_OVERLAP_RATIO,
        chunk_size_limit: Optional[int] = None,
        tokenizer: Optional[Callable[[str], List]] = None,
        separator: str = " ",
    ) -> None:
        """Init params."""
        self.context_window = context_window
        self.num_output = num_output

        self.chunk_overlap_ratio = chunk_overlap_ratio
        if self.chunk_overlap_ratio > 1.0 or self.chunk_overlap_ratio < 0.0:
            raise ValueError("chunk_overlap_ratio must be a float between 0. and 1.")
        self.chunk_size_limit = chunk_size_limit

        # TODO: make configurable
        self._tokenizer = tokenizer or globals_helper.tokenizer
        self._separator = separator

    @classmethod
    def from_llm_metadata(
        cls,
        llm_metadata: LLMMetadata,
        chunk_overlap_ratio: float = DEFAULT_CHUNK_OVERLAP_RATIO,
        chunk_size_limit: Optional[int] = None,
        tokenizer: Optional[Callable[[str], List]] = None,
        separator: str = " ",
    ) -> "PromptHelper":
        """Create from llm predictor.

        This will autofill values like context_window and num_output.

        """
        context_window = llm_metadata.context_window
        if llm_metadata.num_output == -1:
            num_output = DEFAULT_NUM_OUTPUTS
        else:
            num_output = llm_metadata.num_output

        return cls(
            context_window=context_window,
            num_output=num_output,
            chunk_overlap_ratio=chunk_overlap_ratio,
            chunk_size_limit=chunk_size_limit,
            tokenizer=tokenizer,
            separator=separator,
        )

    def _get_available_context_size(self, prompt: BasePromptTemplate) -> int:
        """Get available context size.

        This is calculated as:
            available context window = total context window
                - input (partially filled prompt)
                - output (room reserved for response)
        """
        empty_prompt_txt = get_empty_prompt_txt(prompt)
        prompt_tokens = self._tokenizer(empty_prompt_txt)
        num_prompt_tokens = len(prompt_tokens)

        return self.context_window - num_prompt_tokens - self.num_output

    def _get_available_chunk_size(
        self, prompt: BasePromptTemplate, num_chunks: int = 1, padding: int = 5
    ) -> int:
        """Get available chunk size.

        This is calculated as:
            available context window = total context window
                - input (partially filled prompt)
                - output (room reserved for response)

            available chunk size  = available context window  // number_chunks
                - padding

        Note:
        - By default, we use padding of 5 (to save space for formatting needs).
        - The available chunk size is further clamped to chunk_size_limit if specified
        """
        available_context_size = self._get_available_context_size(prompt)

        result = available_context_size // num_chunks
        result -= padding

        if self.chunk_size_limit is not None:
            result = min(result, self.chunk_size_limit)

        return result

    def get_text_splitter_given_prompt(
        self,
        prompt: BasePromptTemplate,
        num_chunks: int = 1,
        padding: int = DEFAULT_PADDING,
    ) -> TokenTextSplitter:
        """Get text splitter configured to maximally pack available context window,
        taking into account of given prompt, and desired number of chunks.
        """
        chunk_size = self._get_available_chunk_size(prompt, num_chunks, padding=padding)
        if chunk_size == 0:
            raise ValueError("Got 0 as available chunk size.")
        chunk_overlap = int(self.chunk_overlap_ratio * chunk_size)
        text_splitter = TokenTextSplitter(
            separator=self._separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=self._tokenizer,
        )
        return text_splitter

    def truncate(
        self,
        prompt: BasePromptTemplate,
        text_chunks: Sequence[str],
        padding: int = DEFAULT_PADDING,
    ) -> List[str]:
        """Truncate text chunks to fit available context window."""
        text_splitter = self.get_text_splitter_given_prompt(
            prompt,
            num_chunks=len(text_chunks),
            padding=padding,
        )
        return [truncate_text(chunk, text_splitter) for chunk in text_chunks]

    def repack(
        self,
        prompt: BasePromptTemplate,
        text_chunks: Sequence[str],
        padding: int = DEFAULT_PADDING,
    ) -> List[str]:
        """Repack text chunks to fit available context window.

        This will combine text chunks into consolidated chunks
        that more fully "pack" the prompt template given the context_window.

        """
        text_splitter = self.get_text_splitter_given_prompt(prompt, padding=padding)
        combined_str = "\n\n".join([c.strip() for c in text_chunks if c.strip()])
        return text_splitter.split_text(combined_str)
