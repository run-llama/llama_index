"""Core abstractions for building an index of GPT data."""

from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json
from pathlib import Path
from gpt_index.file_reader import SimpleDirectoryReader
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, Prompt, LLMChain
from gpt_index.prompts import (
    DEFAULT_SUMMARY_PROMPT, 
    DEFAULT_QUERY_PROMPT, 
    DEFAULT_TEXT_QA_PROMPT,
    DEFAULT_INSERT_PROMPT
)
from gpt_index.utils import get_chunk_size_given_prompt, extract_number_given_response
from gpt_index.text_splitter import TokenTextSplitter

from typing import Dict, List, Optional, Set
import json


MAX_CHUNK_SIZE = 3900
MAX_CHUNK_OVERLAP = 200
NUM_OUTPUTS = 256



@dataclass
class Node(DataClassJsonMixin):
    """A node in the GPT index."""

    text: str
    index: int
    child_indices: Set[int]


@dataclass
class IndexGraph(DataClassJsonMixin):
    all_nodes: Dict[str, Node]
    root_nodes: Dict[str, Node]

    @property
    def size(self) -> int:
        """Get number of nodes."""
        return len(self.all_nodes)

    def get_children(self, parent_node: Optional[Node]) -> Dict[str, Node]:
        """Get nodes given indices."""
        if parent_node is None:
            return self.root_nodes
        else:
            return {i: self.all_nodes[i] for i in parent_node.child_indices}

    def insert_under_parent(self, node: Node, parent_node: Optional[Node]) -> None:
        """Insert under parent node."""
        if node.index in self.all_nodes:
            raise ValueError(
                "Cannot insert a new node with the same index as an existing node."
            )
        if parent_node is None:
            self.root_nodes[node.index] = node
        else:
            parent_node.child_indices.add(node.index)

        self.all_nodes[node.index] = node


def _get_text_from_nodes(nodes: Dict[int, Node]) -> str:
    """Get text from nodes."""
    text = ""
    for node in nodes.values():
        text += node.text
        text += "\n"
    return text


def _get_numbered_text_from_nodes(nodes: Dict[int, Node]) -> str:
    """Get text from nodes in the format of a numbered list."""
    text = ""
    number = 1
    for node in nodes.values():
        text += f"({number}) {' '.join(node.text.splitlines())}"
        text += "\n\n"
        number += 1
    return text


class GPTIndexBuilder:
    """GPT Index builder."""
    # TODO: consolidate with GPTIndex

    def __init__(
        self, 
        num_children: int = 10,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
    ) -> None:
        """Initialize with params."""
        if num_children < 2:
            raise ValueError("Invalid number of children.")
        self.num_children = num_children
        # instantiate LLM
        llm = OpenAI(temperature=0)
        summary_prompt_obj = Prompt(template=summary_prompt, input_variables=["text"])
        self.summary_llm_chain = LLMChain(prompt=summary_prompt_obj, llm=llm)
        chunk_size = get_chunk_size_given_prompt(
            summary_prompt.format(text=""), MAX_CHUNK_SIZE, num_children, NUM_OUTPUTS
        )
        self.text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=chunk_size, 
            chunk_overlap=MAX_CHUNK_OVERLAP // num_children
        )


    def build_from_text(self, text: str) -> IndexGraph:
        """Build from text.

        Returns:
            IndexGraph: graph object consisting of all_nodes, root_nodes
        
        """
        text_chunks = self.text_splitter.split_text(text)

        # instantiate all_nodes from initial text chunks
        all_nodes = [{i: Node(t, i, set())} for i, t in enumerate(text_chunks)]
        root_nodes = self._build_index_from_nodes(all_nodes, all_nodes)
        return IndexGraph(all_nodes, root_nodes)

    def _build_index_from_nodes(
        self, cur_nodes: Dict[int, Node], all_nodes: Dict[int, Node]
    ) -> Dict[int, Node]:
        """Consolidates chunks recursively, in a bottoms-up fashion."""
        cur_index = len(all_nodes)
        new_node_dict = {}
        print(f'building index from nodes: {len(cur_nodes) // self.num_children} chunks')
        for i in range(0, len(cur_nodes), self.num_children):
            print(f'{i}/{len(cur_nodes)}')
            cur_nodes_chunk = cur_nodes[i:i+self.num_children]

            text_chunk = _get_text_from_nodes(cur_nodes_chunk)

            new_summary = self.summary_llm_chain.predict(text=text_chunk)
            print(f'{i}/{len(cur_nodes)}, summary: {new_summary}')
            new_node = Node(new_summary, cur_index, {n.index for n in cur_nodes_chunk})
            new_node_dict[cur_index] = new_node
            cur_index += 1

        all_nodes.update(new_node_dict)

        if len(new_node_dict) <= self.num_children:
            return new_node_dict
        else:
            return self._build_index_from_nodes(new_node_dict, all_nodes)


class GPTIndexInserter:
    """GPT Index inserter."""

    def __init__(
        self, 
        index_graph: IndexGraph,
        num_children: int = 10,
        insert_prompt: str = DEFAULT_INSERT_PROMPT,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    ) -> None:
        """Initialize with params."""
        if num_children < 2:
            raise ValueError("Invalid number of children.")
        self.num_children = num_children
        # instantiate LLM
        llm = OpenAI(temperature=0)

        # summary
        summary_prompt_obj = Prompt(template=summary_prompt, input_variables=["text"])
        self.summary_llm_chain = LLMChain(prompt=summary_prompt_obj, llm=llm)
        chunk_size = get_chunk_size_given_prompt(
            summary_prompt.format(text=""), MAX_CHUNK_SIZE, num_children, NUM_OUTPUTS
        )
        self.text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=chunk_size, 
            chunk_overlap=MAX_CHUNK_OVERLAP // num_children
        )

        # insert
        insert_prompt_obj = Prompt(
            template=insert_prompt, input_variables=["num_chunks", "context_list", "new_chunk"]
        )
        self.insert_llm_chain = LLMChain(prompt=insert_prompt_obj, llm=llm)
        chunk_size = get_chunk_size_given_prompt(
            insert_prompt.format(text=""), MAX_CHUNK_SIZE, num_children, NUM_OUTPUTS
        )
        self.text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=chunk_size, 
            chunk_overlap=MAX_CHUNK_OVERLAP // num_children
        )

    def _insert_under_parent_and_consolidate(
        self, text_chunk: str, parent_node: Optional[Node]
    ) -> None:
        """Insert node under parent and consolidate.

        Consolidation will happen by dividing up child nodes, and creating a new 
        intermediate layer of nodes.

        """
        # perform insertion
        text_node = Node(text_chunk, self.index_graph.size, set())
        self.index_graph.insert_under_parent(text_node, parent_node)

        # if under num_children limit, then we're fine
        if len(self.index_graph.get_children(parent_node)) <= self.num_children:
            return
        else:
            # perform consolidation
            cur_graph_nodes = self.get_children(parent_node)
            # this layer is all leaf nodes, consolidate and split leaf nodes
            cur_node_index = self.index_graph.size
            # consolidate and split leaf nodes in half
            half1 = cur_graph_nodes[:len(cur_graph_nodes)//2]
            half2 = cur_graph_nodes[len(cur_graph_nodes)//2:]

            text_chunk1 = _get_text_from_nodes(half1)
            summary1 = self.summary_llm_chain.predict(text=text_chunk1)
            node1 = Node(summary1, cur_node_index, {n.index for n in half1})

            text_chunk2 = _get_text_from_nodes(half2)
            summary2 = self.summary_llm_chain.predict(text=text_chunk2)
            node2 = Node(summary2, cur_node_index+1, {n.index for n in half2})
            
            # insert half1 and half2 as new children of parent_node
            # first remove child indices from parent node
            if parent_node is not None:
                parent_node.child_indices = {}
            else:
                self.index_graph.root_nodes = set()
            self.index_graph.insert_under_parent(node1, parent_node)
            self.index_graph.insert_under_parent(node2, parent_node)

    def _insert_node(
        self, text_chunk: str, parent_node: Optional[Node]
    ) -> None:
        """Insert node."""

        cur_graph_nodes = self.get_children(parent_node)
        # check if leaf nodes, then just insert under parent
        if len(cur_graph_nodes[0].child_indices) == 0:
            self._insert_under_parent_and_consolidate(text_chunk, parent_node)
        # else try to find the right summary node to insert under
        else:
            response = self.insert_llm_chain.predict(
                new_chunk_text=text_chunk,
                num_chunks=len(cur_graph_nodes),
                context_list=_get_numbered_text_from_nodes(cur_graph_nodes)
            )
            number = extract_number_given_response(response)
            if number is None:
                # NOTE: if we can't extract a number, then we just insert under parent
                self._insert_under_parent_and_consolidate(text_chunk, parent_node)
            elif number > len(cur_graph_nodes):
                # NOTE: if number is out of range, then we just insert under parent
                self._insert_under_parent_and_consolidate(text_chunk, parent_node)
            selected_node = cur_graph_nodes[number-1]
            self._insert_node(text_chunk, selected_node)

        # now we need to update summary for parent node, since we
        # need to bubble updated summaries up the tree
        if parent_node is not None:
            text_chunk = _get_text_from_nodes(self.index_graph.get_children(parent_node))
            summary = self.summary_llm_chain.predict(text=text_chunk)
            parent_node.text = summary

    def insert(self, text: str) -> None:
        """Insert into index_graph."""
        text_chunks = self.text_splitter.split_text(text)

        for text_chunk in text_chunks:
            self._insert_node(text_chunk, None)
        



@dataclass
class GPTIndex(DataClassJsonMixin):
    """GPT Index."""

    graph: IndexGraph
    query_template: str = DEFAULT_QUERY_PROMPT
    text_qa_template: str = DEFAULT_TEXT_QA_PROMPT
    summary_template: str = DEFAULT_SUMMARY_PROMPT
    num_children: int = 10

    def _query(self, cur_nodes: Dict[int, Node], query_str: str, verbose: bool = False) -> str:
        """Answer a query recursively."""
        query_prompt = Prompt(
            template=self.query_template, 
            input_variables=["num_chunks", "context_list", "query_str"]
        )
        llm = OpenAI(temperature=0)
        llm_chain = LLMChain(prompt=query_prompt, llm=llm) 
        response = llm_chain.predict(
            query_str=query_str,
            num_chunks=len(cur_nodes), 
            context_list=_get_numbered_text_from_nodes(cur_nodes)
        )
        
        if verbose:
            formatted_query = self.query_template.format(
                num_chunks=len(cur_nodes),
                query_str=query_str,
                context_list=_get_numbered_text_from_nodes(cur_nodes)
            )
            print(f'==============')
            print(f'current prompt template: {formatted_query}')
            print(f'cur query response: {response}')
        number = extract_number_given_response(response)
        if number is None:
            print(f"Could not retrieve response - no numbers present")
            # just join text from current nodes as response
            return _get_text_from_nodes(cur_nodes)
        elif number > len(cur_nodes):
            print(f'Invalid response: {response} - number {number} out of range')
            return response

        # number is 1-indexed, so subtract 1
        selected_node = cur_nodes[number-1]
        if len(selected_node.child_indices) == 0:
            answer_prompt = Prompt(
                template=self.text_qa_template, 
                input_variables=["context_str", "query_str"]
            )
            llm_chain = LLMChain(prompt=answer_prompt, llm=llm) 
            response = llm_chain.predict(
                context_str=selected_node.text,
                query_str=query_str
            )
            return response
        else:
            return self._query(
                self.graph.get_children(selected_node), query_str
            )

    def query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        if verbose:
            print('Starting query: {query_str}')
        return self._query(self.graph.root_nodes, query_str, verbose=verbose).strip()
            
    @classmethod
    def from_input_dir(
        cls, 
        input_dir: str,
        index_builder: GPTIndexBuilder = GPTIndexBuilder()
    ) -> "GPTIndex":
        """Builds an index from an input directory.

        Uses the default index builder.
        
        """
        input_dir = Path(input_dir)
        # instantiate file reader
        reader = SimpleDirectoryReader(input_dir)
        text_data = reader.load_data()

        # Use index builder
        index_graph = index_builder.build_from_text(text_data)
        return cls(index_graph)

    @classmethod
    def load_from_disk(cls, save_path: str) -> None:
        """Load from disk."""
        with open(save_path, "r") as f:
            return cls.from_dict(json.load(f))

    def save_to_disk(self, save_path: str) -> None:
        """Safe to file."""
        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f)



if __name__ == "__main__":
    print('hello world')
        
        