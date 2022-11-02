"""Core abstractions for building an index of GPT data."""

from pathlib import Path
from gbt_db_retrieve.file_reader import SimpleDirectoryReader
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, Prompt, LLMChain


from typing import List


class Node:
    """A node in the GPT index."""
    def __init__(self, text: str, index: int, child_indices: List[int]) -> None:
        self.text = text
        self.index = index
        self.child_indices = child_indices


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


class GPTIndex:
    """GPT Index."""

    def __init__(self, text_data: str, num_children: int = 10) -> None:
        """Initialize params."""
        # instantiate LLM
        llm = OpenAI(temperature=0)
        _summary_prompt = (
            "Write a concise summary of the following:\n",
            "\n",
            "\n",
            "{text}\n",
            "\n",
            "\n",
            "CONCISE SUMMARY:\"\"\"\n",
        )
        summary_prompt = Prompt(template=_summary_prompt, input_variables=["text"])
        self.llm_chain = LLMChain(prompt=summary_prompt, llm=self.llm)

        text_splitter = CharacterTextSplitter()
        text_chunks = text_splitter.split_text(text_data)
        # instantiate all_nodes from initial text chunks
        self.all_nodes = [Node(t, i, []) for i, t in enumerate(text_chunks)]
        self.num_children = num_children
        self.root_nodes = self._build_index_from_nodes(self.all_nodes, self.all_nodes)
    

    def _build_index_from_nodes(self, cur_nodes: List[Node], all_nodes: List[Node]) -> List[Node]:
        """Consolidates chunks recursively, in a bottoms-up fashion."""
        cur_index = len(all_nodes)
        new_node_list = []
        for i in range(0, len(cur_nodes), self.num_children):
            cur_nodes_chunk = cur_nodes[i:i+self.num_children]
            text_chunk = _get_text_from_nodes(cur_nodes_chunk)
            new_summary = self.llm_chain.predict(text=text_chunk)
            new_node = Node(new_summary, cur_index, [n.index for n in cur_nodes_chunk])
            new_node_list.append(new_node)
            cur_index += 1

        all_nodes.extend(new_node_list)

        if len(new_node_list) <= self.num_children:
            return new_node_list
        else:
            return self._build_index_from_nodes(new_node_list, all_nodes)


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
        
        template = (
            "We also provide context information below. The full context is provided in a numbered list (1 to {num_chunks}),"
            "where each item in the list corresponds to a chunk of text.\n"
            "---------------------\n"
            "{context_list}"
            "---------------------\n"
            f"Given the context information, we'd like to answer the question: {query_str}\n"
            "Given the question, please choose the number (1 to {num_chunks}) corresponding to the context chunk "
            "that is most relevant to the question.\n"
            "Provide the answer in the following format: \"ANSWER: <number>\"."
        )
        query_prompt = Prompt(template=template, input_variables=["question"])

        return self._query(self.root_nodes, query_prompt)

            

    @classmethod
    def build_index_from_dir(cls, input_dir: Path) -> "GPTIndex":
        """Builds an index from an input directory."""
        # instantiate file reader
        reader = SimpleDirectoryReader(input_dir)
        text_data = reader.load_data()

        return cls(text_data)


        
        