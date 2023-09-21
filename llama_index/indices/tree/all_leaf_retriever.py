"""Summarize query."""

import logging
from typing import Any, List, cast

from llama_index.data_structs.data_structs import IndexGraph
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.tree.base import TreeIndex
from llama_index.indices.utils import get_sorted_node_list
from llama_index.schema import NodeWithScore

logger = logging.getLogger(__name__)

DEFAULT_NUM_CHILDREN = 10


class TreeAllLeafRetriever(BaseRetriever):
    """GPT all leaf retriever.

    This class builds a query-specific tree from leaf nodes to return a response.
    Using this query mode means that the tree index doesn't need to be built
    when initialized, since we rebuild the tree for each query.

    Args:
        text_qa_template (Optional[BasePromptTemplate]): Question-Answer Prompt
            (see :ref:`Prompt-Templates`).

    """

    def __init__(self, index: TreeIndex, **kwargs: Any) -> None:
        self._index = index
        self._index_struct = index.index_struct
        self._docstore = index.docstore

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
