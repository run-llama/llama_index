"""Leaf query mechanism."""

import logging
from typing import Any, Dict, List, Optional, cast

from langchain.input import print_text

from gpt_index.data_structs.data_structs_v2 import IndexGraph
from gpt_index.data_structs.node_v2 import Node
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.builder import ResponseBuilder
from gpt_index.indices.utils import extract_numbers_given_response, get_sorted_node_list
from gpt_index.prompts.default_prompts import (
    DEFAULT_QUERY_PROMPT,
    DEFAULT_QUERY_PROMPT_MULTIPLE,
)
from gpt_index.prompts.prompts import TreeSelectMultiplePrompt, TreeSelectPrompt
from gpt_index.response.schema import Response

logger = logging.getLogger(__name__)


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

        if len(self.index_struct.get_children(selected_node)) == 0:
            response_builder = ResponseBuilder(
                self._service_context,
                self.text_qa_template,
                self.refine_template,
            )
            self.response_builder.add_node_as_source(selected_node)
            # use response builder to get answer from node
            node_text = self._get_text_from_node(selected_node, level=level)
            cur_response = response_builder.get_response_over_chunks(
                query_str, [node_text], prev_response=prev_response
            )
            cur_response = cast(str, cur_response)
            logger.debug(f">[Level {level}] Current answer response: {cur_response} ")
        else:
            cur_response = self._query_level(
                self.index_struct.get_children(selected_node),
                query_bundle,
                level=level + 1,
            )

        if prev_response is None:
            return cur_response
        else:
            context_msg = selected_node.get_text()
            (
                cur_response,
                formatted_refine_prompt,
            ) = self._service_context.llm_predictor.predict(
                self.refine_template,
                query_str=query_str,
                existing_answer=prev_response,
                context_msg=context_msg,
            )

            logger.debug(f">[Level {level}] Refine prompt: {formatted_refine_prompt}")
            logger.debug(f">[Level {level}] Current refined response: {cur_response} ")
            return cur_response

    def _query_level(
        self,
        cur_node_ids: Dict[int, str],
        query_bundle: QueryBundle,
        level: int = 0,
    ) -> str:
        """Answer a query recursively."""
        query_str = query_bundle.query_str
        cur_nodes = {
            index: self._docstore.get_node(node_id)
            for index, node_id in cur_node_ids.items()
        }
        cur_node_list = get_sorted_node_list(cur_nodes)

        if len(cur_node_list) == 1:
            logger.debug(f">[Level {level}] Only one node left. Querying node.")
            return self._query_with_selected_node(
                cur_node_list[0], query_bundle, level=level
            )
        elif self.child_branch_factor == 1:
            query_template = self.query_template.partial_format(
                num_chunks=len(cur_node_list), query_str=query_str
            )
            numbered_node_text = (
                self._service_context.prompt_helper.get_numbered_text_from_nodes(
                    cur_node_list, prompt=query_template
                )
            )
            (
                response,
                formatted_query_prompt,
            ) = self._service_context.llm_predictor.predict(
                query_template,
                context_list=numbered_node_text,
            )
        else:
            query_template_multiple = self.query_template_multiple.partial_format(
                num_chunks=len(cur_node_list),
                query_str=query_str,
                branching_factor=self.child_branch_factor,
            )
            numbered_node_text = (
                self._service_context.prompt_helper.get_numbered_text_from_nodes(
                    cur_node_list, prompt=query_template_multiple
                )
            )
            (
                response,
                formatted_query_prompt,
            ) = self._service_context.llm_predictor.predict(
                query_template_multiple,
                context_list=numbered_node_text,
            )

        logger.debug(
            f">[Level {level}] current prompt template: {formatted_query_prompt}"
        )
        self._service_context.llama_logger.add_log(
            {"formatted_prompt_template": formatted_query_prompt, "level": level}
        )
        debug_str = f">[Level {level}] Current response: {response}"
        logger.debug(debug_str)
        if self._verbose:
            print_text(debug_str, end="\n")

        numbers = extract_numbers_given_response(response, n=self.child_branch_factor)
        if numbers is None:
            debug_str = (
                f">[Level {level}] Could not retrieve response - no numbers present"
            )
            logger.debug(debug_str)
            if self._verbose:
                print_text(debug_str, end="\n")
            # just join text from current nodes as response
            return response
        result_response = None
        for number_str in numbers:
            number = int(number_str)
            if number > len(cur_node_list):
                logger.debug(
                    f">[Level {level}] Invalid response: {response} - "
                    f"number {number} out of range"
                )
                return response

            # number is 1-indexed, so subtract 1
            selected_node = cur_node_list[number - 1]

            info_str = (
                f">[Level {level}] Selected node: "
                f"[{number}]/[{','.join([str(int(n)) for n in numbers])}]"
            )
            logger.info(info_str)
            if self._verbose:
                print_text(info_str, end="\n")
            debug_str = " ".join(selected_node.get_text().splitlines())
            full_debug_str = (
                f">[Level {level}] Node "
                f"[{number}] Summary text: "
                f"{ selected_node.get_text() }"
            )
            logger.debug(full_debug_str)
            if self._verbose:
                print_text(full_debug_str, end="\n")
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
        info_str = f"> Starting query: {query_bundle.query_str}"
        logger.info(info_str)
        if self._verbose:
            print_text(info_str, end="\n")
        response_str = self._query_level(
            self.index_struct.root_nodes,
            query_bundle,
            level=0,
        ).strip()
        return Response(response_str, source_nodes=self.response_builder.get_sources())

    def _select_nodes(
        self,
        cur_node_list: List[Node],
        query_bundle: QueryBundle,
        level: int = 0,
    ) -> List[Node]:
        query_str = query_bundle.query_str

        if self.child_branch_factor == 1:
            query_template = self.query_template.partial_format(
                num_chunks=len(cur_node_list), query_str=query_str
            )
            numbered_node_text = (
                self._service_context.prompt_helper.get_numbered_text_from_nodes(
                    cur_node_list, prompt=query_template
                )
            )
            (
                response,
                formatted_query_prompt,
            ) = self._service_context.llm_predictor.predict(
                query_template,
                context_list=numbered_node_text,
            )
        else:
            query_template_multiple = self.query_template_multiple.partial_format(
                num_chunks=len(cur_node_list),
                query_str=query_str,
                branching_factor=self.child_branch_factor,
            )
            numbered_node_text = (
                self._service_context.prompt_helper.get_numbered_text_from_nodes(
                    cur_node_list, prompt=query_template_multiple
                )
            )
            (
                response,
                formatted_query_prompt,
            ) = self._service_context.llm_predictor.predict(
                query_template_multiple,
                context_list=numbered_node_text,
            )

        logger.debug(
            f">[Level {level}] current prompt template: {formatted_query_prompt}"
        )
        self._service_context.llama_logger.add_log(
            {"formatted_prompt_template": formatted_query_prompt, "level": level}
        )
        debug_str = f">[Level {level}] Current response: {response}"
        logger.debug(debug_str)
        if self._verbose:
            print_text(debug_str, end="\n")

        numbers = extract_numbers_given_response(response, n=self.child_branch_factor)
        if numbers is None:
            debug_str = (
                f">[Level {level}] Could not retrieve response - no numbers present"
            )
            logger.debug(debug_str)
            if self._verbose:
                print_text(debug_str, end="\n")
            # just join text from current nodes as response
            return []

        selected_nodes = []
        for number_str in numbers:
            number = int(number_str)
            if number > len(cur_node_list):
                logger.debug(
                    f">[Level {level}] Invalid response: {response} - "
                    f"number {number} out of range"
                )
                continue

            # number is 1-indexed, so subtract 1
            selected_node = cur_node_list[number - 1]

            info_str = (
                f">[Level {level}] Selected node: "
                f"[{number}]/[{','.join([str(int(n)) for n in numbers])}]"
            )
            logger.info(info_str)
            if self._verbose:
                print_text(info_str, end="\n")
            debug_str = " ".join(selected_node.get_text().splitlines())
            full_debug_str = (
                f">[Level {level}] Node "
                f"[{number}] Summary text: "
                f"{ selected_node.get_text() }"
            )
            logger.debug(full_debug_str)
            if self._verbose:
                print_text(full_debug_str, end="\n")
            selected_nodes.append(selected_node)

        return selected_nodes

    def _retrieve_level(
        self,
        cur_node_ids: Dict[int, str],
        query_bundle: QueryBundle,
        level: int = 0,
    ) -> List[Node]:
        """Answer a query recursively."""
        cur_nodes = {
            index: self._docstore.get_node(node_id)
            for index, node_id in cur_node_ids.items()
        }
        cur_node_list = get_sorted_node_list(cur_nodes)

        if len(cur_node_list) > self.child_branch_factor:
            selected_nodes = self._select_nodes(
                cur_node_list,
                query_bundle,
                level=level,
            )
        else:
            selected_nodes = cur_node_list

        children_nodes = {}
        for node in selected_nodes:
            node_dict = self.index_struct.get_children(node)
            children_nodes.update(node_dict)

        if len(children_nodes) == 0:
            # NOTE: leaf level
            return selected_nodes
        else:
            return self._retrieve_level(children_nodes, query_bundle, level + 1)

    def _retrieve(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        return self._retrieve_level(
            self.index_struct.root_nodes,
            query_bundle,
            level=0,
        )
