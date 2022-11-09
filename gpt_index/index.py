"""Core abstractions for building an index of GPT data."""

from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json
from pathlib import Path
from gpt_index.file_reader import SimpleDirectoryReader
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, Prompt, LLMChain
from gpt_index.prompts import DEFAULT_SUMMARY_PROMPT, DEFAULT_QUERY_PROMPT, DEFAULT_TEXT_QA_PROMPT
from gpt_index.utils import get_chunk_size_given_prompt, extract_number_given_response
from gpt_index.text_splitter import TokenTextSplitter

from typing import List, Dict, Set
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
    all_nodes: Dict[int, Node]
    root_nodes: Dict[int, Node]

    @property
    def size(self):
        return len(self.all_nodes)


def _get_sorted_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    sorted_indices = sorted(node_dict.keys())
    return [node_dict[index] for index in sorted_indices]


def _get_text_from_nodes(node_list: List[Node]) -> str:
    """Get text from nodes."""
    text = ""
    for node in node_list:
        text += node.text
        text += "\n"
    return text


def _get_numbered_text_from_nodes(node_list: List[Node]) -> str:
    """Get text from nodes in the format of a numbered list."""
    text = ""
    number = 1
    for node in node_list:
        text += f"({number}) {' '.join(node.text.splitlines())}"
        text += "\n\n"
        number += 1
    return text


class GPTIndexBuilder:
    """GPT Index builder."""

    def __init__(
        self, 
        num_children: int = 10,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    ) -> None:
        """Initialize with params."""
        self.num_children = num_children
        # instantiate LLM
        summary_prompt_obj = Prompt(template=summary_prompt, input_variables=["text"])
        llm = OpenAI(temperature=0)
        self.llm_chain = LLMChain(prompt=summary_prompt_obj, llm=llm)
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
        all_nodes = {i: Node(t, i, set()) for i, t in enumerate(text_chunks)}
        root_nodes = self._build_index_from_nodes(all_nodes, all_nodes)
        return IndexGraph(all_nodes, root_nodes)

    def _build_index_from_nodes(
        self, cur_nodes: Dict[int, Node], all_nodes: Dict[int, Node]
    ) -> Dict[int, Node]:
        """Consolidates chunks recursively, in a bottoms-up fashion."""
        cur_node_list = _get_sorted_node_list(cur_nodes)
        cur_index = len(all_nodes)
        new_node_dict = {}
        print(f'> Building index from nodes: {len(cur_nodes) // self.num_children} chunks')
        for i in range(0, len(cur_node_list), self.num_children):
            print(f'{i}/{len(cur_nodes)}')
            cur_nodes_chunk = cur_node_list[i:i+self.num_children]

            text_chunk = _get_text_from_nodes(cur_nodes_chunk)

            new_summary = self.llm_chain.predict(text=text_chunk)
            print(f'> {i}/{len(cur_nodes)}, summary: {new_summary}')
            new_node = Node(
                new_summary, cur_index, {n.index for n in cur_nodes_chunk}
            )
            new_node_dict[cur_index] = new_node
            cur_index += 1

        all_nodes.update(new_node_dict)

        if len(new_node_dict) <= self.num_children:
            return new_node_dict
        else:
            return self._build_index_from_nodes(new_node_dict, all_nodes)


@dataclass
class GPTIndex(DataClassJsonMixin):
    """GPT Index."""

    graph: IndexGraph
    query_template: str = DEFAULT_QUERY_PROMPT
    text_qa_template: str = DEFAULT_TEXT_QA_PROMPT

    def _query(self, cur_nodes: Dict[int, Node], query_str: str, verbose: bool = False) -> str:
        """Answer a query recursively."""
        cur_node_list = _get_sorted_node_list(cur_nodes)
        query_prompt = Prompt(
            template=self.query_template, 
            input_variables=["num_chunks", "context_list", "query_str"]
        )
        llm = OpenAI(temperature=0)
        llm_chain = LLMChain(prompt=query_prompt, llm=llm) 
        response = llm_chain.predict(
            query_str=query_str,
            num_chunks=len(cur_node_list), 
            context_list=_get_numbered_text_from_nodes(cur_node_list)
        )
        
        if verbose:
            formatted_query = self.query_template.format(
                num_chunks=len(cur_node_list),
                query_str=query_str,
                context_list=_get_numbered_text_from_nodes(cur_node_list)
            )
            print(f'==============')
            print(f'> current prompt template: {formatted_query}')
        number = extract_number_given_response(response)
        if number is None:
            if verbose:
                print(f"> Could not retrieve response - no numbers present")
            # just join text from current nodes as response
            return response
        elif number > len(cur_node_list):
            if verbose:
                print(f'> Invalid response: {response} - number {number} out of range')
            return response

        # number is 1-indexed, so subtract 1
        selected_node = cur_node_list[number-1]
        print(f"> Selected node: {response}")
        print(f"> Node Summary text: {' '.join(selected_node.text.splitlines())}")

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
            if verbose:
                formatted_answer_prompt = self.text_qa_template.format(
                    context_str=selected_node.text,
                    query_str=query_str
                )
                print('==============')
                print(f'> final answer prompt: {formatted_answer_prompt}')
            return response
        else:
            return self._query(
                {i: self.graph.all_nodes[i] for i in selected_node.child_indices}, 
                query_str,
                verbose=verbose
            )

    def query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f'> Starting query: {query_str}')
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
            return cls(graph=IndexGraph.from_dict(json.load(f)))

    def save_to_disk(self, save_path: str) -> None:
        """Safe to file."""
        with open(save_path, "w") as f:
            json.dump(self.graph.to_dict(), f)

        
        