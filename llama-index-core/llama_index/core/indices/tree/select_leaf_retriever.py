"""Leaf query mechanism."""

import logging
from typing import Any, Dict, List, Optional, cast

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.response.schema import Response
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.indices.tree.base import TreeIndex
from llama_index.core.indices.tree.utils import get_numbered_text_from_nodes
from llama_index.core.indices.utils import (
    extract_numbers_given_response,
    get_sorted_node_list,
)
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_REFINE_PROMPT_SEL,
)
from llama_index.core.prompts.default_prompts import (
    DEFAULT_QUERY_PROMPT,
    DEFAULT_QUERY_PROMPT_MULTIPLE,
    DEFAULT_TEXT_QA_PROMPT,
)
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeWithScore,
    QueryBundle,
)
from llama_index.core.settings import Settings
from llama_index.core.utils import print_text, truncate_text

logger = logging.getLogger(__name__)


def get_text_from_node(
    node: BaseNode,
    level: Optional[int] = None,
    verbose: bool = False,
) -> str:
    """Get text from node."""
    level_str = "" if level is None else f"[Level {level}]"
    fmt_text_chunk = truncate_text(node.get_content(metadata_mode=MetadataMode.LLM), 50)
    logger.debug(f">{level_str} Searching in chunk: {fmt_text_chunk}")

    response_txt = node.get_content(metadata_mode=MetadataMode.LLM)
    fmt_response = truncate_text(response_txt, 200)
    if verbose:
        print_text(f">{level_str} Got node text: {fmt_response}\n", color="blue")
    return response_txt


class TreeSelectLeafRetriever(BaseRetriever):
    """
    Tree select leaf retriever.

    This class traverses the index graph and searches for a leaf node that can best
    answer the query.

    Args:
        query_template (Optional[BasePromptTemplate]): Tree Select Query Prompt
            (see :ref:`Prompt-Templates`).
        query_template_multiple (Optional[BasePromptTemplate]): Tree Select
            Query Prompt (Multiple)
            (see :ref:`Prompt-Templates`).
        child_branch_factor (int): Number of child nodes to consider at each level.
            If child_branch_factor is 1, then the query will only choose one child node
            to traverse for any given parent node.
            If child_branch_factor is 2, then the query will choose two child nodes.

    """

    def __init__(
        self,
        index: TreeIndex,
        query_template: Optional[BasePromptTemplate] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        refine_template: Optional[BasePromptTemplate] = None,
        query_template_multiple: Optional[BasePromptTemplate] = None,
        child_branch_factor: int = 1,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        **kwargs: Any,
    ):
        self._index = index
        self._llm = index._llm
        self._index_struct = index.index_struct
        self._docstore = index.docstore
        self._prompt_helper = Settings._prompt_helper or PromptHelper.from_llm_metadata(
            self._llm.metadata,
        )

        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self._refine_template = refine_template or DEFAULT_REFINE_PROMPT_SEL
        self.query_template = query_template or DEFAULT_QUERY_PROMPT
        self.query_template_multiple = (
            query_template_multiple or DEFAULT_QUERY_PROMPT_MULTIPLE
        )
        self.child_branch_factor = child_branch_factor
        super().__init__(
            callback_manager=callback_manager or Settings.callback_manager,
            object_map=object_map,
            verbose=verbose,
        )

    def _query_with_selected_node(
        self,
        selected_node: BaseNode,
        query_bundle: QueryBundle,
        prev_response: Optional[str] = None,
        level: int = 0,
    ) -> str:
        """
        Get response for selected node.

        If not leaf node, it will recursively call _query on the child nodes.
        If prev_response is provided, we will update prev_response with the answer.

        """
        query_str = query_bundle.query_str

        if len(self._index_struct.get_children(selected_node)) == 0:
            response_builder = get_response_synthesizer(
                llm=self._llm,
                text_qa_template=self._text_qa_template,
                refine_template=self._refine_template,
                callback_manager=self.callback_manager,
            )
            # use response builder to get answer from node
            node_text = get_text_from_node(selected_node, level=level)
            cur_response = response_builder.get_response(
                query_str, [node_text], prev_response=prev_response
            )
            cur_response = str(cur_response)
            logger.debug(f">[Level {level}] Current answer response: {cur_response} ")
        else:
            cur_response = self._query_level(
                self._index_struct.get_children(selected_node),
                query_bundle,
                level=level + 1,
            )

        if prev_response is None:
            return cur_response
        else:
            context_msg = selected_node.get_content(metadata_mode=MetadataMode.LLM)
            cur_response = self._llm.predict(
                self._refine_template,
                query_str=query_str,
                existing_answer=prev_response,
                context_msg=context_msg,
            )

            logger.debug(f">[Level {level}] Current refined response: {cur_response} ")
            return str(cur_response)

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
            text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
                prompt=query_template,
                num_chunks=len(cur_node_list),
                llm=self._llm,
            )
            numbered_node_text = get_numbered_text_from_nodes(
                cur_node_list, text_splitter=text_splitter
            )

            response = self._llm.predict(
                query_template,
                context_list=numbered_node_text,
            )
        else:
            query_template_multiple = self.query_template_multiple.partial_format(
                num_chunks=len(cur_node_list),
                query_str=query_str,
                branching_factor=self.child_branch_factor,
            )

            text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
                prompt=query_template_multiple,
                num_chunks=len(cur_node_list),
                llm=self._llm,
            )
            numbered_node_text = get_numbered_text_from_nodes(
                cur_node_list, text_splitter=text_splitter
            )

            response = self._llm.predict(
                query_template_multiple,
                context_list=numbered_node_text,
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
            debug_str = " ".join(
                selected_node.get_content(metadata_mode=MetadataMode.LLM).splitlines()
            )
            full_debug_str = (
                f">[Level {level}] Node "
                f"[{number}] Summary text: "
                f"{selected_node.get_content(metadata_mode=MetadataMode.LLM)}"
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
            self._index_struct.root_nodes,
            query_bundle,
            level=0,
        ).strip()
        # TODO: fix source nodes
        return Response(response_str, source_nodes=[])

    def _select_nodes(
        self,
        cur_node_list: List[BaseNode],
        query_bundle: QueryBundle,
        level: int = 0,
    ) -> List[BaseNode]:
        query_str = query_bundle.query_str

        if self.child_branch_factor == 1:
            query_template = self.query_template.partial_format(
                num_chunks=len(cur_node_list), query_str=query_str
            )
            text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
                prompt=query_template,
                num_chunks=len(cur_node_list),
                llm=self._llm,
            )
            numbered_node_text = get_numbered_text_from_nodes(
                cur_node_list, text_splitter=text_splitter
            )

            response = self._llm.predict(
                query_template,
                context_list=numbered_node_text,
            )
        else:
            query_template_multiple = self.query_template_multiple.partial_format(
                num_chunks=len(cur_node_list),
                query_str=query_str,
                branching_factor=self.child_branch_factor,
            )

            text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
                prompt=query_template_multiple,
                num_chunks=len(cur_node_list),
                llm=self._llm,
            )
            numbered_node_text = get_numbered_text_from_nodes(
                cur_node_list, text_splitter=text_splitter
            )

            response = self._llm.predict(
                query_template_multiple,
                context_list=numbered_node_text,
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
            debug_str = " ".join(
                selected_node.get_content(metadata_mode=MetadataMode.LLM).splitlines()
            )
            full_debug_str = (
                f">[Level {level}] Node "
                f"[{number}] Summary text: "
                f"{selected_node.get_content(metadata_mode=MetadataMode.LLM)}"
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
    ) -> List[BaseNode]:
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
            node_dict = self._index_struct.get_children(node)
            children_nodes.update(node_dict)

        if len(children_nodes) == 0:
            # NOTE: leaf level
            return selected_nodes
        else:
            return self._retrieve_level(children_nodes, query_bundle, level + 1)

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Get nodes for response."""
        nodes = self._retrieve_level(
            self._index_struct.root_nodes,
            query_bundle,
            level=0,
        )
        return [NodeWithScore(node=node) for node in nodes]
