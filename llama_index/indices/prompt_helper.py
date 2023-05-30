"""General prompt helper that can help deal with token limitations.

The helper can split text. It can also concatenate text from Node
structs but keeping token limitations in mind.

"""

from typing import Callable, List, Optional, Sequence

from llama_index.constants import MAX_CHUNK_OVERLAP
from llama_index.data_structs.node import Node
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.llm_predictor.base import BaseLLMPredictor
from llama_index.prompts.base import Prompt
from llama_index.prompts.utils import get_empty_prompt_txt
from llama_index.utils import globals_helper


class PromptHelper:
    """Prompt helper.

    This utility helps us fill in the prompt, split the text,
    and fill in context information according to necessary token limitations.

    Args:
        max_input_size (int): Maximum input size for the LLM.
        num_output (int): Number of outputs for the LLM.
        max_chunk_overlap (int): Maximum chunk overlap for the LLM.
        embedding_limit (Optional[int]): Maximum number of embeddings to use.
        chunk_size_limit (Optional[int]): Maximum chunk size to use.
        tokenizer (Optional[Callable[[str], List]]): Tokenizer to use.

    """

    def __init__(
        self,
        max_input_size: int,
        num_output: int,
        max_chunk_overlap: int,
        embedding_limit: Optional[int] = None,
        chunk_size_limit: Optional[int] = None,
        tokenizer: Optional[Callable[[str], List]] = None,
        separator: str = " ",
    ) -> None:
        """Init params."""
        self.max_input_size = max_input_size
        self.num_output = num_output
        self.max_chunk_overlap = max_chunk_overlap
        self.embedding_limit = embedding_limit
        self.chunk_size_limit = chunk_size_limit
        # TODO: make configurable
        self._tokenizer = tokenizer or globals_helper.tokenizer
        self._separator = separator
        self.use_chunk_size_limit = chunk_size_limit is not None

    @classmethod
    def from_llm_predictor(
        self,
        llm_predictor: BaseLLMPredictor,
        max_chunk_overlap: Optional[int] = None,
        embedding_limit: Optional[int] = None,
        chunk_size_limit: Optional[int] = None,
        tokenizer: Optional[Callable[[str], List]] = None,
    ) -> "PromptHelper":
        """Create from llm predictor.

        This will autofill values like max_input_size and num_output.

        """
        llm_metadata = llm_predictor.get_llm_metadata()
        max_chunk_overlap = max_chunk_overlap or min(
            MAX_CHUNK_OVERLAP,
            llm_metadata.max_input_size // 10,
        )
        if chunk_size_limit is not None:
            max_chunk_overlap = min(max_chunk_overlap, chunk_size_limit // 10)

        return self(
            llm_metadata.max_input_size,
            llm_metadata.num_output,
            max_chunk_overlap,
            embedding_limit=embedding_limit,
            chunk_size_limit=chunk_size_limit,
            tokenizer=tokenizer,
        )

    def _get_chunk_size_given_prompt(
        self, prompt_text: str, num_chunks: int, padding: Optional[int] = 5
    ) -> int:
        """Get chunk size making sure we can also fit the prompt in.

        Chunk size is computed based on a function of the total input size,
        the prompt length, the number of outputs, and the number of chunks.

        If padding is specified, then we subtract that from the chunk size.
        By default we assume there is a padding of 5 (for the newline between chunks
        and other formatting needs).

        Limit by embedding_limit and chunk_size_limit if specified.

        """
        prompt_tokens = self._tokenizer(prompt_text)
        num_prompt_tokens = len(prompt_tokens)

        # NOTE: if embedding limit is specified, then chunk_size must not be larger than
        # embedding_limit
        result = (
            self.max_input_size - num_prompt_tokens - self.num_output
        ) // num_chunks
        if padding is not None:
            result -= padding

        if self.embedding_limit is not None:
            result = min(result, self.embedding_limit)
        if self.chunk_size_limit is not None and self.use_chunk_size_limit:
            result = min(result, self.chunk_size_limit)

        return result

    def get_text_splitter_given_prompt(
        self, prompt: Prompt, num_chunks: int, padding: Optional[int] = 5
    ) -> TokenTextSplitter:
        """Get text splitter given initial prompt.

        Allows us to get the text splitter which will split up text according
        to the desired chunk size.

        """
        # generate empty_prompt_txt to compute initial tokens
        empty_prompt_txt = get_empty_prompt_txt(prompt)
        chunk_size = self._get_chunk_size_given_prompt(
            empty_prompt_txt, num_chunks, padding=padding
        )
        text_splitter = TokenTextSplitter(
            separator=self._separator,
            chunk_size=chunk_size,
            chunk_overlap=self.max_chunk_overlap // num_chunks,
            tokenizer=self._tokenizer,
        )
        return text_splitter

    def get_text_from_nodes(
        self, node_list: List[Node], prompt: Optional[Prompt] = None
    ) -> str:
        """Get text from nodes. Used by tree-structured indices."""
        num_nodes = len(node_list)
        text_splitter = None
        if prompt is not None:
            # add padding given the newline character
            text_splitter = self.get_text_splitter_given_prompt(
                prompt,
                num_nodes,
                padding=1,
            )
        results = []
        for node in node_list:
            text = (
                text_splitter.truncate_text(node.get_text())
                if text_splitter is not None
                else node.get_text()
            )
            results.append(text)

        return "\n".join(results)

    def compact_text_chunks(
        self, prompt: Prompt, text_chunks: Sequence[str]
    ) -> List[str]:
        """Compact text chunks.

        This will combine text chunks into consolidated chunks
        that more fully "pack" the prompt template given the max_input_size.

        """
        combined_str = "\n\n".join([c.strip() for c in text_chunks if c.strip()])
        # resplit based on self.max_chunk_overlap
        text_splitter = self.get_text_splitter_given_prompt(prompt, 1, padding=1)
        return text_splitter.split_text(combined_str)
