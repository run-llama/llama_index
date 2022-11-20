"""Leaf query mechanism."""

from typing import Dict, Optional, cast

from gpt_index.indices.base import BaseGPTIndexQuery
from gpt_index.indices.data_structs import IndexGraph, Node
from gpt_index.indices.utils import (
    extract_numbers_given_response,
    get_numbered_text_from_nodes,
    get_sorted_node_list,
)
from gpt_index.langchain_helpers.chain_wrapper import openai_llm_predict
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import (
    DEFAULT_QUERY_PROMPT,
    DEFAULT_QUERY_PROMPT_MULTIPLE,
    DEFAULT_REFINE_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)


class GPTTreeIndexLeafQuery(BaseGPTIndexQuery[IndexGraph]):
    """GPT Tree Index leaf query.

    This class traverses the index graph and searches for a leaf node that can best
    answer the query.

    """

    def __init__(
        self,
        index_struct: IndexGraph,
        query_template: Prompt = DEFAULT_QUERY_PROMPT,
        query_template_multiple: Prompt = DEFAULT_QUERY_PROMPT_MULTIPLE,
        text_qa_template: Prompt = DEFAULT_TEXT_QA_PROMPT,
        refine_template: Prompt = DEFAULT_REFINE_PROMPT,
        child_branch_factor: int = 1,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct)
        self.query_template = query_template
        self.query_template_multiple = query_template_multiple
        self.text_qa_template = text_qa_template
        self.refine_template = refine_template
        self.child_branch_factor = child_branch_factor

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
