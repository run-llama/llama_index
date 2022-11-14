"""Tree-based index."""

import json
from typing import Any, Dict, List, Optional, cast

from langchain import LLMChain, OpenAI, Prompt

from gpt_index.constants import MAX_CHUNK_OVERLAP, MAX_CHUNK_SIZE, NUM_OUTPUTS
from gpt_index.indices.base import BaseGPTIndex
from gpt_index.indices.data_structs import IndexGraph, Node
from gpt_index.indices.utils import (
    extract_numbers_given_response,
    get_chunk_size_given_prompt,
    get_numbered_text_from_nodes,
    get_sorted_node_list,
    get_text_from_nodes,
)
from gpt_index.langchain_helpers.chain_wrapper import openai_llm_predict
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts import (
    DEFAULT_QUERY_PROMPT,
    DEFAULT_QUERY_PROMPT_MULTIPLE,
    DEFAULT_REFINE_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)
from gpt_index.schema import Document


class GPTTreeIndexBuilder:
    """GPT tree index builder.

    Helper class to build the tree-structured index.

    """

    def __init__(
        self, num_children: int = 10, summary_prompt: str = DEFAULT_SUMMARY_PROMPT
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

            new_summary = self.llm_chain.predict(text=text_chunk)
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
        summary_template: str = DEFAULT_SUMMARY_PROMPT,
        query_template: str = DEFAULT_QUERY_PROMPT,
        query_template_multiple: str = DEFAULT_QUERY_PROMPT_MULTIPLE,
        text_qa_template: str = DEFAULT_TEXT_QA_PROMPT,
        refine_template: str = DEFAULT_REFINE_PROMPT,
        num_children: int = 10,
        child_branch_factor: int = 1,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.summary_template = summary_template
        self.query_template = query_template
        self.query_template_multiple = query_template_multiple
        self.text_qa_template = text_qa_template
        self.refine_template = refine_template
        self.num_children = num_children
        self.child_branch_factor = child_branch_factor
        super().__init__(documents=documents, index_struct=index_struct)

    def _query_with_selected_node(
        self,
        selected_node: Node,
        query_str: str,
        prev_response: Optional[str] = None,
        level: int = 0,
        verbose: bool = False,
    ) -> str:
        """Get response for selected node.

        If not leaf node, it will recursively call _query on the child nodes.
        If prev_response is provided, we will update prev_response with the answer.

        """
        if len(selected_node.child_indices) == 0:
            cur_response, formatted_answer_prompt = openai_llm_predict(
                self.text_qa_template,
                context_str=selected_node.text,
                query_str=query_str,
            )
            if verbose:
                print(f">[Level {level}] answer prompt: {formatted_answer_prompt}")
            print(f">[Level {level}] Current answer response: {cur_response} ")
        else:
            cur_response = self._query(
                {
                    i: self.index_struct.all_nodes[i]
                    for i in selected_node.child_indices
                },
                query_str,
                level=level + 1,
                verbose=verbose,
            )

        if prev_response is None:
            return cur_response
        else:
            context_msg = "\n".join([selected_node.text, cur_response])
            cur_response, formatted_refine_prompt = openai_llm_predict(
                self.refine_template,
                query_str=query_str,
                existing_answer=prev_response,
                context_msg=context_msg,
            )

            if verbose:
                print(f">[Level {level}] Refine prompt: {formatted_refine_prompt}")
            print(f">[Level {level}] Current refined response: {cur_response} ")
            return cur_response

    def _query(
        self,
        cur_nodes: Dict[int, Node],
        query_str: str,
        level: int = 0,
        verbose: bool = False,
    ) -> str:
        """Answer a query recursively."""
        cur_node_list = get_sorted_node_list(cur_nodes)

        if self.child_branch_factor == 1:
            response, formatted_query_prompt = openai_llm_predict(
                self.query_template,
                num_chunks=len(cur_node_list),
                query_str=query_str,
                context_list=get_numbered_text_from_nodes(cur_node_list),
            )
        else:
            response, formatted_query_prompt = openai_llm_predict(
                self.query_template_multiple,
                num_chunks=len(cur_node_list),
                query_str=query_str,
                context_list=get_numbered_text_from_nodes(cur_node_list),
                branching_factor=self.child_branch_factor,
            )

        if verbose:
            print(f">[Level {level}] current prompt template: {formatted_query_prompt}")

        numbers = extract_numbers_given_response(response, n=self.child_branch_factor)
        if numbers is None:
            if verbose:
                print(
                    f">[Level {level}] Could not retrieve response - no numbers present"
                )
            # just join text from current nodes as response
            return response
        result_response = None
        for number_str in numbers:
            number = int(number_str)
            if number > len(cur_node_list):
                if verbose:
                    print(
                        f">[Level {level}] Invalid response: {response} - "
                        f"number {number} out of range"
                    )
                return response

            # number is 1-indexed, so subtract 1
            selected_node = cur_node_list[number - 1]
            print(
                f">[Level {level}] Selected node: "
                f"[{number}]/[{','.join([str(int(n)) for n in numbers])}]"
            )
            print(
                f">[Level {level}] Node "
                f"[{number}] Summary text: {' '.join(selected_node.text.splitlines())}"
            )
            result_response = self._query_with_selected_node(
                selected_node,
                query_str,
                prev_response=result_response,
                level=level,
                verbose=verbose,
            )
        # result_response should not be None
        return cast(str, result_response)

    def query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")
        return self._query(
            self.index_struct.root_nodes, query_str, level=0, verbose=verbose
        ).strip()

    def build_index_from_documents(self, documents: List[Document]) -> IndexGraph:
        """Build the index from documents."""
        # do simple concatenation
        text_data = "\n".join([d.text for d in documents])
        index_builder = GPTTreeIndexBuilder(self.num_children, self.summary_template)
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
