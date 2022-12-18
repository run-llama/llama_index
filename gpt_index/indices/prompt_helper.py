"""General prompt helper that can help deal with token limitations.

The helper can split text. It can also concatenate text from Node
structs but keeping token limitations in mind.

"""

from typing import Callable, List, Optional

from gpt_index.constants import MAX_CHUNK_OVERLAP, MAX_CHUNK_SIZE, NUM_OUTPUTS
from gpt_index.indices.data_structs import Node
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.base import Prompt
from gpt_index.utils import globals_helper


class PromptHelper:
    """Prompt helper."""

    def __init__(
        self,
        max_input_size: int = MAX_CHUNK_SIZE,
        num_output: int = NUM_OUTPUTS,
        max_chunk_overlap: int = MAX_CHUNK_OVERLAP,
        embedding_limit: Optional[int] = None,
        tokenizer: Optional[Callable[[str], List]] = None,
    ) -> None:
        """Init params."""
        self.max_input_size = max_input_size
        self.num_output = num_output
        self.max_chunk_overlap = max_chunk_overlap
        self.embedding_limit = embedding_limit
        # TODO: make configurable
        self._tokenizer = tokenizer or globals_helper.tokenizer

    def get_chunk_size_given_prompt(
        self, prompt_text: str, num_chunks: int, padding: Optional[int] = 1
    ) -> int:
        """Get chunk size making sure we can also fit the prompt in.

        Chunk size is computed based on a function of the total input size,
        the prompt length, the number of outputs, and the number of chunks.

        If padding is specified, then we subtract that from the chunk size.
        By default we assume there is a padding of 1 (for the newline between chunks).

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
            return min(result, self.embedding_limit)
        else:
            return result

    def get_text_splitter_given_prompt(
        self, prompt: Prompt, num_chunks: int, padding: Optional[int] = 1
    ) -> TokenTextSplitter:
        """Get text splitter given initial prompt.

        Allows us to get the text splitter which will split up text according
        to the desired chunk size.

        """
        # generate empty_prompt_txt to compute initial tokens
        fmt_dict = {v: "" for v in prompt.input_variables}
        empty_prompt_txt = prompt.format(**fmt_dict)
        chunk_size = self.get_chunk_size_given_prompt(
            empty_prompt_txt, num_chunks, padding=padding
        )
        text_splitter = TokenTextSplitter(
            separator=" ",
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

    def get_numbered_text_from_nodes(
        self, node_list: List[Node], prompt: Optional[Prompt] = None
    ) -> str:
        """Get text from nodes in the format of a numbered list.

        Used by tree-structured indices.

        """
        num_nodes = len(node_list)
        text_splitter = None
        if prompt is not None:
            # add padding given the number, and the newlines
            text_splitter = self.get_text_splitter_given_prompt(
                prompt,
                num_nodes,
                padding=5,
            )
        results = []
        number = 1
        for node in node_list:
            node_text = " ".join(node.get_text().splitlines())
            if text_splitter is not None:
                node_text = text_splitter.truncate_text(node_text)
            text = f"({number}) {node_text}"
            results.append(text)
            number += 1
        return "\n\n".join(results)
