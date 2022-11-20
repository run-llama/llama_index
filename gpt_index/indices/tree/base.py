"""Tree-based index."""

import json
from typing import Any, Dict, List, Optional

from gpt_index.constants import MAX_CHUNK_OVERLAP, MAX_CHUNK_SIZE, NUM_OUTPUTS
from gpt_index.indices.base import DEFAULT_MODE, BaseGPTIndex, BaseGPTIndexQuery
from gpt_index.indices.data_structs import IndexGraph, Node
from gpt_index.indices.tree.leaf_query import GPTTreeIndexLeafQuery
from gpt_index.indices.tree.retrieve_query import GPTTreeIndexRetQuery
from gpt_index.indices.utils import (
    get_chunk_size_given_prompt,
    get_sorted_node_list,
    get_text_from_nodes,
)
from gpt_index.langchain_helpers.chain_wrapper import openai_llm_predict
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.base import Prompt, validate_prompt
from gpt_index.prompts.default_prompts import DEFAULT_SUMMARY_PROMPT
from gpt_index.schema import Document

RETRIEVE_MODE = "retrieve"


class GPTTreeIndexBuilder:
    """GPT tree index builder.

    Helper class to build the tree-structured index.

    """

    def __init__(
        self, num_children: int = 10, summary_prompt: Prompt = DEFAULT_SUMMARY_PROMPT
    ) -> None:
        """Initialize with params."""
        self.num_children = num_children
        self.summary_prompt = summary_prompt
        chunk_size = get_chunk_size_given_prompt(
            summary_prompt.format(text=""), MAX_CHUNK_SIZE, num_children, NUM_OUTPUTS
        )
        self.text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=MAX_CHUNK_OVERLAP // num_children,
        )

    def build_from_text(self, text: str) -> IndexGraph:
        """Build from text.

        Returns:
            IndexGraph: graph object consisting of all_nodes, root_nodes

        """
        text_chunks = self.text_splitter.split_text(text)

        # instantiate all_nodes from initial text chunks
        all_nodes = {i: Node(t, i, set()) for i, t in enumerate(text_chunks)}
        root_nodes = self._build_index_from_nodes(all_nodes, all_nodes)
        return IndexGraph(all_nodes, root_nodes)

    def _build_index_from_nodes(
        self, cur_nodes: Dict[int, Node], all_nodes: Dict[int, Node]
    ) -> Dict[int, Node]:
        """Consolidates chunks recursively, in a bottoms-up fashion."""
        cur_node_list = get_sorted_node_list(cur_nodes)
        cur_index = len(all_nodes)
        new_node_dict = {}
        print(
            f"> Building index from nodes: {len(cur_nodes) // self.num_children} chunks"
        )
        for i in range(0, len(cur_node_list), self.num_children):
            print(f"{i}/{len(cur_nodes)}")
            cur_nodes_chunk = cur_node_list[i : i + self.num_children]
            text_chunk = get_text_from_nodes(cur_nodes_chunk)

            new_summary, _ = openai_llm_predict(self.summary_prompt, text=text_chunk)

            print(f"> {i}/{len(cur_nodes)}, summary: {new_summary}")
            new_node = Node(new_summary, cur_index, {n.index for n in cur_nodes_chunk})
            new_node_dict[cur_index] = new_node
            cur_index += 1

        all_nodes.update(new_node_dict)

        if len(new_node_dict) <= self.num_children:
            return new_node_dict
        else:
            return self._build_index_from_nodes(new_node_dict, all_nodes)


class GPTTreeIndex(BaseGPTIndex[IndexGraph]):
    """GPT Index."""

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        index_struct: Optional[IndexGraph] = None,
        summary_template: Prompt = DEFAULT_SUMMARY_PROMPT,
        query_str: Optional[str] = None,
        num_children: int = 10,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.num_children = num_children
        # if query_str is specified, then we try to load into summary template
        if query_str is not None:
            summary_template = summary_template.partial_format(query_str=query_str)
        self.summary_template = summary_template
        validate_prompt(self.summary_template, ["text"], ["query_str"])
        super().__init__(documents=documents, index_struct=index_struct)

    def _mode_to_query(self, mode: str, **query_kwargs: Any) -> BaseGPTIndexQuery:
        """Query mode to class."""
        if mode == DEFAULT_MODE:
            query: BaseGPTIndexQuery = GPTTreeIndexLeafQuery(
                self.index_struct, **query_kwargs
            )
        elif mode == RETRIEVE_MODE:
            query = GPTTreeIndexRetQuery(self.index_struct, **query_kwargs)
        else:
            raise ValueError(f"Invalid query mode: {mode}.")
        return query

    def build_index_from_documents(self, documents: List[Document]) -> IndexGraph:
        """Build the index from documents."""
        # do simple concatenation
        text_data = "\n".join([d.text for d in documents])
        index_builder = GPTTreeIndexBuilder(
            num_children=self.num_children, summary_prompt=self.summary_template
        )
        index_graph = index_builder.build_from_text(text_data)
        return index_graph

    @classmethod
    def load_from_disk(cls, save_path: str, **kwargs: Any) -> "GPTTreeIndex":
        """Load from disk."""
        with open(save_path, "r") as f:
            return cls(index_struct=IndexGraph.from_dict(json.load(f)), **kwargs)

    def save_to_disk(self, save_path: str) -> None:
        """Safe to file."""
        with open(save_path, "w") as f:
            json.dump(self.index_struct.to_dict(), f)
