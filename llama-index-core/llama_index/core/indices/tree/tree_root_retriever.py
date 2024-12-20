"""Retrieve query."""
import logging
from typing import Any, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.indices.tree.base import TreeIndex
from llama_index.core.indices.utils import get_sorted_node_list
from llama_index.core.schema import NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)


class TreeRootRetriever(BaseRetriever):
    """Tree root retriever.

    This class directly retrieves the answer from the root nodes.

    Unlike GPTTreeIndexLeafQuery, this class assumes the graph already stores
    the answer (because it was constructed with a query_str), so it does not
    attempt to parse information down the graph in order to synthesize an answer.
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
        root_nodes = self._docstore.get_node_dict(self._index_struct.root_nodes)
        sorted_nodes = get_sorted_node_list(root_nodes)
        return [NodeWithScore(node=node) for node in sorted_nodes]
