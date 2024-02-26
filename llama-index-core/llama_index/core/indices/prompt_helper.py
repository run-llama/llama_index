"""General prompt helper that can help deal with LLM context window token limitations.

At its core, it calculates available context size by starting with the context window
size of an LLM and reserve token space for the prompt template, and the output.

It provides utility for "repacking" text chunks (retrieved from index) to maximally
make use of the available context window (and thereby reducing the number of LLM calls
needed), or truncating them so that they fit in a single LLM call.
"""

import logging
from copy import deepcopy
from string import Formatter
from typing import Callable, List, Optional, Sequence

from llama_index.core.base.llms.types import ChatMessage, LLMMetadata
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser.text.token import TokenTextSplitter
from llama_index.core.node_parser.text.utils import truncate_text
from llama_index.core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    SelectorPromptTemplate,
)
from llama_index.core.prompts.prompt_utils import get_empty_prompt_txt
from llama_index.core.schema import BaseComponent
from llama_index.core.utilities.token_counting import TokenCounter

DEFAULT_PADDING = 5
DEFAULT_CHUNK_OVERLAP_RATIO = 0.1

logger = logging.getLogger(__name__)


class PromptHelper(BaseComponent):
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

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum context size that will get sent to the LLM.",
    )
    num_output: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The amount of token-space to leave in input for generation.",
    )
    chunk_overlap_ratio: float = Field(
        default=DEFAULT_CHUNK_OVERLAP_RATIO,
        description="The percentage token amount that each chunk should overlap.",
    )
    chunk_size_limit: Optional[int] = Field(description="The maximum size of a chunk.")
    separator: str = Field(
        default=" ", description="The separator when chunking tokens."
    )

    _token_counter: TokenCounter = PrivateAttr()

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
        if chunk_overlap_ratio > 1.0 or chunk_overlap_ratio < 0.0:
            raise ValueError("chunk_overlap_ratio must be a float between 0. and 1.")

        # TODO: make configurable
        self._token_counter = TokenCounter(tokenizer=tokenizer)

        super().__init__(
            context_window=context_window,
            num_output=num_output,
            chunk_overlap_ratio=chunk_overlap_ratio,
            chunk_size_limit=chunk_size_limit,
            separator=separator,
        )

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

    @classmethod
    def class_name(cls) -> str:
        return "PromptHelper"

    def _get_available_context_size(self, num_prompt_tokens: int) -> int:
        """Get available context size.

        This is calculated as:
            available context window = total context window
                - input (partially filled prompt)
                - output (room reserved for response)

        Notes:
        - Available context size is further clamped to be non-negative.
        """
        context_size_tokens = self.context_window - num_prompt_tokens - self.num_output
        if context_size_tokens < 0:
            raise ValueError(
                f"Calculated available context size {context_size_tokens} was"
                " not non-negative."
            )
        return context_size_tokens

    def _get_available_chunk_size(
        self,
        prompt: BasePromptTemplate,
        num_chunks: int = 1,
        padding: int = 5,
        llm: Optional[LLM] = None,
    ) -> int:
        """Get available chunk size.

        This is calculated as:
            available chunk size = available context window  // number_chunks
                - padding

        Notes:
        - By default, we use padding of 5 (to save space for formatting needs).
        - Available chunk size is further clamped to chunk_size_limit if specified.
        """
        if isinstance(prompt, SelectorPromptTemplate):
            prompt = prompt.select(llm=llm)

        if isinstance(prompt, ChatPromptTemplate):
            messages: List[ChatMessage] = prompt.message_templates

            # account for partial formatting
            partial_messages = []
            for message in messages:
                partial_message = deepcopy(message)

                # get string variables (if any)
                template_vars = [
                    var
                    for _, var, _, _ in Formatter().parse(str(message))
                    if var is not None
                ]

                # figure out which variables are partially formatted
                # if a variable is not formatted, it will be replaced with
                # the template variable itself
                used_vars = {
                    template_var: f"{{{template_var}}}"
                    for template_var in template_vars
                }
                for var_name, val in prompt.kwargs.items():
                    if var_name in template_vars:
                        used_vars[var_name] = val

                # format partial message
                if partial_message.content is not None:
                    partial_message.content = partial_message.content.format(
                        **used_vars
                    )

                # add to list of partial messages
                partial_messages.append(partial_message)

            num_prompt_tokens = self._token_counter.estimate_tokens_in_messages(
                partial_messages
            )
        else:
            prompt_str = get_empty_prompt_txt(prompt)
            num_prompt_tokens = self._token_counter.get_string_tokens(prompt_str)

        available_context_size = self._get_available_context_size(num_prompt_tokens)
        result = available_context_size // num_chunks - padding
        if self.chunk_size_limit is not None:
            result = min(result, self.chunk_size_limit)
        return result

    def get_text_splitter_given_prompt(
        self,
        prompt: BasePromptTemplate,
        num_chunks: int = 1,
        padding: int = DEFAULT_PADDING,
        llm: Optional[LLM] = None,
    ) -> TokenTextSplitter:
        """Get text splitter configured to maximally pack available context window,
        taking into account of given prompt, and desired number of chunks.
        """
        chunk_size = self._get_available_chunk_size(
            prompt, num_chunks, padding=padding, llm=llm
        )
        if chunk_size <= 0:
            raise ValueError(f"Chunk size {chunk_size} is not positive.")
        chunk_overlap = int(self.chunk_overlap_ratio * chunk_size)
        return TokenTextSplitter(
            separator=self.separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=self._token_counter.tokenizer,
        )

    def truncate(
        self,
        prompt: BasePromptTemplate,
        text_chunks: Sequence[str],
        padding: int = DEFAULT_PADDING,
        llm: Optional[LLM] = None,
    ) -> List[str]:
        """Truncate text chunks to fit available context window."""
        text_splitter = self.get_text_splitter_given_prompt(
            prompt,
            num_chunks=len(text_chunks),
            padding=padding,
            llm=llm,
        )
        return [truncate_text(chunk, text_splitter) for chunk in text_chunks]

    def repack(
        self,
        prompt: BasePromptTemplate,
        text_chunks: Sequence[str],
        padding: int = DEFAULT_PADDING,
        llm: Optional[LLM] = None,
    ) -> List[str]:
        """Repack text chunks to fit available context window.

        This will combine text chunks into consolidated chunks
        that more fully "pack" the prompt template given the context_window.

        """
        text_splitter = self.get_text_splitter_given_prompt(
            prompt, padding=padding, llm=llm
        )
        combined_str = "\n\n".join([c.strip() for c in text_chunks if c.strip()])
        return text_splitter.split_text(combined_str)
