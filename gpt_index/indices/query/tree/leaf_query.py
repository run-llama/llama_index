"""Leaf query mechanism."""

import logging
from typing import Any, Dict, Optional, cast

from gpt_index.data_structs.data_structs import IndexGraph, Node
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.builder import ResponseBuilder
from gpt_index.indices.utils import (
    extract_numbers_given_response,
    get_sorted_node_list,
    truncate_text,
)
from gpt_index.prompts.default_prompts import (
    DEFAULT_QUERY_PROMPT,
    DEFAULT_QUERY_PROMPT_MULTIPLE,
)
from gpt_index.prompts.prompts import TreeSelectMultiplePrompt, TreeSelectPrompt
from gpt_index.response.schema import Response


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
        child_branch_factor: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct, **kwargs)
        self.query_template = query_template or DEFAULT_QUERY_PROMPT
        self.query_template_multiple = (
            query_template_multiple or DEFAULT_QUERY_PROMPT_MULTIPLE
        )
        self.child_branch_factor = child_branch_factor

    def _query_with_selected_node(
        self,
        selected_node: Node,
        query_bundle: QueryBundle,
        prev_response: Optional[str] = None,
        level: int = 0,
    ) -> str:
        """Get response for selected node.

        If not leaf node, it will recursively call _query on the child nodes.
        If prev_response is provided, we will update prev_response with the answer.

        """
        query_str = query_bundle.query_str

        if len(selected_node.child_indices) == 0:
            response_builder = ResponseBuilder(
                self._prompt_helper,
                self._llm_predictor,
                self.text_qa_template,
                self.refine_template,
            )
            self.response_builder.add_node_as_source(selected_node)
            # use response builder to get answer from node
            node_text, _ = self._get_text_from_node(
                query_bundle, selected_node, level=level
            )
            cur_response = response_builder.get_response_over_chunks(
                query_str, [node_text], prev_response=prev_response
            )
            logging.debug(f">[Level {level}] Current answer response: {cur_response} ")
        else:
            cur_response = self._query_level(
                {
                    i: self.index_struct.all_nodes[i]
                    for i in selected_node.child_indices
                },
                query_bundle,
                level=level + 1,
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

            logging.debug(f">[Level {level}] Refine prompt: {formatted_refine_prompt}")
            logging.debug(f">[Level {level}] Current refined response: {cur_response} ")
            return cur_response

    def _query_level(
        self,
        cur_nodes: Dict[int, Node],
        query_bundle: QueryBundle,
        level: int = 0,
    ) -> str:
        """Answer a query recursively."""
        query_str = query_bundle.query_str
        cur_node_list = get_sorted_node_list(cur_nodes)

        if len(cur_node_list) == 1:
            logging.debug(f">[Level {level}] Only one node left. Querying node.")
            return self._query_with_selected_node(
                cur_node_list[0], query_bundle, level=level
            )
        elif self.child_branch_factor == 1:
            query_template = self.query_template.partial_format(
                num_chunks=len(cur_node_list), query_str=query_str
            )
            numbered_node_text = self._prompt_helper.get_numbered_text_from_nodes(
                cur_node_list, prompt=query_template
            )
            response, formatted_query_prompt = self._llm_predictor.predict(
                query_template,
                context_list=numbered_node_text,
            )
        else:
            query_template_multiple = self.query_template_multiple.partial_format(
                num_chunks=len(cur_node_list),
                query_str=query_str,
                branching_factor=self.child_branch_factor,
            )
            numbered_node_text = self._prompt_helper.get_numbered_text_from_nodes(
                cur_node_list, prompt=query_template_multiple
            )
            response, formatted_query_prompt = self._llm_predictor.predict(
                query_template_multiple,
                context_list=numbered_node_text,
            )

        logging.debug(
            f">[Level {level}] current prompt template: {formatted_query_prompt}"
        )

        numbers = extract_numbers_given_response(response, n=self.child_branch_factor)
        if numbers is None:
            logging.debug(
                f">[Level {level}] Could not retrieve response - no numbers present"
            )
            # just join text from current nodes as response
            return response
        result_response = None
        for number_str in numbers:
            number = int(number_str)
            if number > len(cur_node_list):
                logging.debug(
                    f">[Level {level}] Invalid response: {response} - "
                    f"number {number} out of range"
                )
                return response

            # number is 1-indexed, so subtract 1
            selected_node = cur_node_list[number - 1]

            logging.info(
                f">[Level {level}] Selected node: "
                f"[{number}]/[{','.join([str(int(n)) for n in numbers])}]"
            )
            debug_str = " ".join(selected_node.get_text().splitlines())
            logging.debug(
                f">[Level {level}] Node "
                f"[{number}] Summary text: "
                f"{ truncate_text(debug_str, 100) }"
            )
            result_response = self._query_with_selected_node(
                selected_node,
                query_bundle,
                prev_response=result_response,
                level=level,
            )
        # result_response should not be None
        return cast(str, result_response)

    def _query(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        # NOTE: this overrides the _query method in the base class
        logging.info(f"> Starting query: {query_bundle.query_str}")
        response_str = self._query_level(
            self.index_struct.root_nodes,
            query_bundle,
            level=0,
        ).strip()
        return Response(response_str, source_nodes=self.response_builder.get_sources())
