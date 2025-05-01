"""Summarize query."""

import logging
from typing import Any, List, Optional, cast

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.data_structs.data_structs import IndexGraph
from llama_index.core.indices.tree.base import TreeIndex
from llama_index.core.indices.utils import get_sorted_node_list
from llama_index.core.schema import NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)

DEFAULT_NUM_CHILDREN = 10


class TreeAllLeafRetriever(BaseRetriever):
    """
    GPT all leaf retriever.

    This class builds a query-specific tree from leaf nodes to return a response.
    Using this query mode means that the tree index doesn't need to be built
    when initialized, since we rebuild the tree for each query.

    Args:
        text_qa_template (Optional[BasePromptTemplate]): Question-Answer Prompt
            (see :ref:`Prompt-Templates`).

    """

    def __init__(
        self,
        index: TreeIndex,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self._index = index
        self._index_struct = index.index_struct
        self._docstore = index.docstore
        super().__init__(
            callback_manager=callback_manager, object_map=object_map, verbose=verbose
        )

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Get nodes for response."""
        logger.info(f"> Starting query: {query_bundle.query_str}")
        index_struct = cast(IndexGraph, self._index_struct)
        all_nodes = self._docstore.get_node_dict(index_struct.all_nodes)
        sorted_node_list = get_sorted_node_list(all_nodes)
        return [NodeWithScore(node=node) for node in sorted_node_list]
