"""General prompt helper that can help deal with token limitations.

The helper can split text. It can also concatenate text from Node
structs but keeping token limitations in mind.

"""

from typing import Callable, List, Optional

from gpt_index.constants import MAX_CHUNK_OVERLAP
from gpt_index.data_structs.data_structs import Node
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.base import Prompt
from gpt_index.utils import globals_helper


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
        llm_predictor: LLMPredictor,
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

    def get_chunk_size_given_prompt(
        self, prompt_text: str, num_chunks: int, padding: Optional[int] = 1
    ) -> int:
        """Get chunk size making sure we can also fit the prompt in.

        Chunk size is computed based on a function of the total input size,
        the prompt length, the number of outputs, and the number of chunks.

        If padding is specified, then we subtract that from the chunk size.
        By default we assume there is a padding of 1 (for the newline between chunks).

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

    def _get_empty_prompt_txt(self, prompt: Prompt) -> str:
        """Get empty prompt text.

        Substitute empty strings in parts of the prompt that have
        not yet been filled out. Skip variables that have already
        been partially formatted. This is used to compute the initial tokens.

        """
        fmt_dict = {
            v: "" for v in prompt.input_variables if v not in prompt.partial_dict
        }
        empty_prompt_txt = prompt.format(**fmt_dict)
        return empty_prompt_txt

    def get_biggest_prompt(self, prompts: List[Prompt]) -> Prompt:
        """Get biggest prompt.

        Oftentimes we need to fetch the biggest prompt, in order to
        be the most conservative about chunking text. This
        is a helper utility for that.

        """
        empty_prompt_txts = [self._get_empty_prompt_txt(prompt) for prompt in prompts]
        empty_prompt_txt_lens = [len(txt) for txt in empty_prompt_txts]
        biggest_prompt = prompts[
            empty_prompt_txt_lens.index(max(empty_prompt_txt_lens))
        ]
        return biggest_prompt

    def get_text_splitter_given_prompt(
        self, prompt: Prompt, num_chunks: int, padding: Optional[int] = 1
    ) -> TokenTextSplitter:
        """Get text splitter given initial prompt.

        Allows us to get the text splitter which will split up text according
        to the desired chunk size.

        """
        # generate empty_prompt_txt to compute initial tokens
        empty_prompt_txt = self._get_empty_prompt_txt(prompt)
        chunk_size = self.get_chunk_size_given_prompt(
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

    def compact_text_chunks(self, prompt: Prompt, text_chunks: List[str]) -> List[str]:
        """Compact text chunks.

        This will combine text chunks into consolidated chunks
        that more fully "pack" the prompt template given the max_input_size.

        """
        combined_str = "\n\n".join([c.strip() for c in text_chunks if c.strip()])
        # resplit based on self.max_chunk_overlap
        text_splitter = self.get_text_splitter_given_prompt(prompt, 1, padding=1)
        return text_splitter.split_text(combined_str)
