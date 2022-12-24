"""Leaf query mechanism."""

from typing import Any, Dict, Optional, cast

from gpt_index.indices.data_structs import IndexGraph, Node
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.utils import (
    extract_numbers_given_response,
    get_sorted_node_list,
    truncate_text,
)
from gpt_index.prompts.default_prompts import (
    DEFAULT_QUERY_PROMPT,
    DEFAULT_QUERY_PROMPT_MULTIPLE,
    DEFAULT_REFINE_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)
from gpt_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    TreeSelectMultiplePrompt,
    TreeSelectPrompt,
)


class GPTTreeIndexLeafQuery(BaseGPTIndexQuery[IndexGraph]):
    """GPT Tree Index leaf query.

    This class traverses the index graph and searches for a leaf node that can best
    answer the query.

    .. code-block:: python

        response = index.query("<query_str>", mode="default")

    Args:
        query_template (Optional[TreeSelectPrompt]): Tree Select Query Prompt
            (see :ref:`Prompt-Templates`).
        query_template_multiple (Optional[TreeSelectMultiplePrompt]): Tree Select
            Query Prompt (Multiple)
            (see :ref:`Prompt-Templates`).
        text_qa_template (Optional[QuestionAnswerPrompt]): Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        refine_template (Optional[RefinePrompt]): Refinement Prompt
            (see :ref:`Prompt-Templates`).
        child_branch_factor (int): Number of child nodes to consider at each level.
            If child_branch_factor is 1, then the query will only choose one child node
            to traverse for any given parent node.
            If child_branch_factor is 2, then the query will choose two child nodes.

    """

    def __init__(
        self,
        index_struct: IndexGraph,
        query_template: Optional[TreeSelectPrompt] = None,
        query_template_multiple: Optional[TreeSelectMultiplePrompt] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        child_branch_factor: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct, **kwargs)
        self.query_template = query_template or DEFAULT_QUERY_PROMPT
        self.query_template_multiple = (
            query_template_multiple or DEFAULT_QUERY_PROMPT_MULTIPLE
        )
        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self.refine_template = refine_template or DEFAULT_REFINE_PROMPT
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
            # call _query_node to get an answer from doc (either Document/IndexStruct)
            cur_response = self._query_node(
                query_str,
                selected_node,
                self.text_qa_template,
                self.refine_template,
                verbose=verbose,
                level=level,
            )
            if verbose:
                print(f">[Level {level}] Current answer response: {cur_response} ")
        else:
            cur_response = self._query_level(
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
            context_msg = "\n".join([selected_node.get_text(), cur_response])
            cur_response, formatted_refine_prompt = self._llm_predictor.predict(
                self.refine_template,
                query_str=query_str,
                existing_answer=prev_response,
                context_msg=context_msg,
            )

            if verbose:
                print(f">[Level {level}] Refine prompt: {formatted_refine_prompt}")
                print(f">[Level {level}] Current refined response: {cur_response} ")
            return cur_response

    def _query_level(
        self,
        cur_nodes: Dict[int, Node],
        query_str: str,
        level: int = 0,
        verbose: bool = False,
    ) -> str:
        """Answer a query recursively."""
        cur_node_list = get_sorted_node_list(cur_nodes)

        if self.child_branch_factor == 1:
            numbered_node_text = self._prompt_helper.get_numbered_text_from_nodes(
                cur_node_list, prompt=self.query_template
            )
            response, formatted_query_prompt = self._llm_predictor.predict(
                self.query_template,
                num_chunks=len(cur_node_list),
                query_str=query_str,
                context_list=numbered_node_text,
            )
        else:
            numbered_node_text = self._prompt_helper.get_numbered_text_from_nodes(
                cur_node_list, prompt=self.query_template_multiple
            )
            response, formatted_query_prompt = self._llm_predictor.predict(
                self.query_template_multiple,
                num_chunks=len(cur_node_list),
                query_str=query_str,
                context_list=numbered_node_text,
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
            if verbose:
                summary_text = " ".join(selected_node.get_text().splitlines())
                fmt_summary_text = truncate_text(summary_text, 100)
                print(
                    f">[Level {level}] Node "
                    f"[{number}] Summary text: {fmt_summary_text}"
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

    def _query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")
        return self._query_level(
            self.index_struct.root_nodes, query_str, level=0, verbose=verbose
        ).strip()
