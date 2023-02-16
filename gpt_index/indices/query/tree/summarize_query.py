"""Summarize query."""

import logging
from typing import Any, List, Optional, cast

from gpt_index.data_structs.data_structs import IndexGraph, Node
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.builder import ResponseMode
from gpt_index.indices.utils import get_sorted_node_list


class GPTTreeIndexSummarizeQuery(BaseGPTIndexQuery[IndexGraph]):
    """GPT Tree Index summarize query.

    This class builds a query-specific tree from leaf nodes to return a response.
    Using this query mode means that the tree index doesn't need to be built
    when initialized, since we rebuild the tree for each query.

    .. code-block:: python

        response = index.query("<query_str>", mode="summarize")

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): Question-Answer Prompt
            (see :ref:`Prompt-Templates`).

    """

    def __init__(
        self,
        index_struct: IndexGraph,
        num_children: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if "response_mode" in kwargs:
            raise ValueError(
                "response_mode should not be specified for summarize query"
            )
        response_kwargs = kwargs.pop("response_kwargs", {})
        response_kwargs.update(num_children=num_children)
        super().__init__(
            index_struct,
            response_mode=ResponseMode.TREE_SUMMARIZE,
            response_kwargs=response_kwargs,
            **kwargs,
        )

    def _get_nodes_for_response(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        logging.info(f"> Starting query: {query_bundle.query_str}")
        index_struct = cast(IndexGraph, self._index_struct)
        sorted_node_list = get_sorted_node_list(index_struct.all_nodes)
        return sorted_node_list
