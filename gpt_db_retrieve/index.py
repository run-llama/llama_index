"""Core abstractions for building an index of GPT data."""

from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from pathlib import Path
from gpt_db_retrieve.file_reader import SimpleDirectoryReader
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, Prompt, LLMChain
from gpt_db_retrieve.prompts import DEFAULT_SUMMARY_PROMPT, DEFAULT_QUERY_PROMPT
from gpt_db_retrieve.utils import get_chunk_size_given_prompt
from gpt_db_retrieve.text_splitter import TokenTextSplitter

from typing import List, Tuple


MAX_CHUNK_SIZE = 3900
MAX_CHUNK_OVERLAP = 200
NUM_OUTPUTS = 256


@dataclass
class Node(DataClassJsonMixin):
    """A node in the GPT index."""

    text: str
    index: int
    child_indices: List[int]


@dataclass
class IndexGraph(DataClassJsonMixin):
    all_nodes: List[Node]
    root_nodes: List[Node]


def _get_text_from_nodes(nodes: List[Node]) -> str:
    """Get text from nodes."""
    text = ""
    for node in nodes:
        text += node.text
        text += "\n"
    return text


def _get_numbered_text_from_nodes(nodes: List[Node]) -> str:
    """Get text from nodes in the format of a numbered list."""
    text = ""
    number = 1
    for node in nodes:
        text += f"{number}. {node.text}"
        text += "\n\n"
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
        llm = OpenAI(temperature=0)
        summary_prompt_obj = Prompt(template=summary_prompt, input_variables=["text"])
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
        all_nodes = [Node(t, i, []) for i, t in enumerate(text_chunks)]
        root_nodes = self._build_index_from_nodes(all_nodes, all_nodes)
        return all_nodes, root_nodes

    def _build_index_from_nodes(
        self, cur_nodes: List[Node], all_nodes: List[Node]
    ) -> List[Node]:
        """Consolidates chunks recursively, in a bottoms-up fashion."""
        cur_index = len(all_nodes)
        new_node_list = []
        print(f'building index from nodes: {len(cur_nodes) // self.num_children} chunks')
        for i in range(0, len(cur_nodes), self.num_children):
            print(f'{i}/{len(cur_nodes)}')
            cur_nodes_chunk = cur_nodes[i:i+self.num_children]

            text_chunk = _get_text_from_nodes(cur_nodes_chunk)

            new_summary = self.llm_chain.predict(text=text_chunk)
            print(f'{i}/{len(cur_nodes)}, summary: {new_summary}')
            new_node = Node(new_summary, cur_index, [n.index for n in cur_nodes_chunk])
            new_node_list.append(new_node)
            cur_index += 1

        all_nodes.extend(new_node_list)

        if len(new_node_list) <= self.num_children:
            return new_node_list
        else:
            return self._build_index_from_nodes(new_node_list, all_nodes)


class GPTIndex:
    """GPT Index."""

    def __init__(
        self, graph: IndexGraph, query_template: str = DEFAULT_QUERY_PROMPT
    ) -> None:
        """Initialize params."""
        self.graph = graph
        self.query_template = query_template

    def _query(self, cur_nodes: List[Node], query_prompt: Prompt) -> str:
        """Answer a query recursively."""
        response = self.llm_chain.predict(
            num_chunks=len(cur_nodes), context_list=_get_numbered_text_from_nodes(cur_nodes)
        )
        try:
            number = int(response.split(":")[1].strip())
        except Exception as e:
            print(f"Exception occured: {e}, could not retrieve response.")
            # just join text from current nodes as response
            return _get_text_from_nodes(cur_nodes)

        # number is 1-indexed, so subtract 1
        selected_node = cur_nodes[number-1]
        if len(selected_node.child_indices) == 0:
            return selected_node.text
        else:
            return self._query([self.all_nodes[i] for i in selected_node.child_indices], query_prompt)

    def query(self, query_str: str) -> str:
        """Answer a query."""
        query_prompt = Prompt(
            template=self.query_template.format(query_str=query_str), 
            input_variables=["num_chunks", "context_list"]
        )

        return self._query(self.graph.root_nodes, query_prompt)
            
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

if __name__ == "__main__":
    print('hello world')
        
        