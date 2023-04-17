"""Summarize query."""

import logging
from typing import Any, Dict, List, Optional, cast


from gpt_index.data_structs.data_structs_v2 import IndexGraph
from gpt_index.data_structs.node_v2 import Node
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.response_builder import ResponseMode
from gpt_index.indices.utils import get_sorted_node_list

logger = logging.getLogger(__name__)

DEFAULT_NUM_CHILDREN = 10


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
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            index_struct,
            **kwargs,
        )

    @classmethod
    def from_args(  # type: ignore
        cls,
        response_mode: ResponseMode = ResponseMode.TREE_SUMMARIZE,
        response_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> BaseGPTIndexQuery:
        if response_mode != ResponseMode.TREE_SUMMARIZE:
            raise ValueError(
                "response_mode should not be specified for summarize query"
            )
        response_kwargs = kwargs.pop("response_kwargs", {})
        assert isinstance(response_kwargs, dict)
        response_kwargs.update(
            num_children=kwargs.pop("num_children", DEFAULT_NUM_CHILDREN)
        )

        return super().from_args(
            response_mode=response_mode,
            response_kwargs=response_kwargs,
            **kwargs,
        )

    def _retrieve(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        logger.info(f"> Starting query: {query_bundle.query_str}")
        index_struct = cast(IndexGraph, self._index_struct)
        all_nodes = self._docstore.get_node_dict(index_struct.all_nodes)
        sorted_node_list = get_sorted_node_list(all_nodes)
        return sorted_node_list
