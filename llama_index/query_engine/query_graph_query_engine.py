from typing import Dict, List, Optional, Tuple

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.indices.composability.graph import ComposableGraph
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.schema import TextNode, IndexNode, NodeWithScore


class QueryGraphQueryEngine(BaseQueryEngine):
    """Query graph query engine.

    This query engine can operate over a query graph.
    It can take in query engines for its sub-indices.

    If a query engine is a RetrieverQueryEngine, then it will
    first retrieve nodes. If any of the nodes are IndexNodes,
    then it will recursively query the query engine referenced by
    the index node. It will then synthesize the nodes.

    If the query engine is not a RetrieverQueryEngine, then it will
    simply query the query engine.

    Args:
        root_id (str): The root id of the query graph.
        query_engines (Optional[Dict[str, BaseQueryEngine]]): A dictionary of
            id to query engines.

    """

    def __init__(
        self,
        root_id: str,
        query_engine_dict: Dict[str, BaseQueryEngine],
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Init params."""
        self._root_id = root_id
        self._query_engine_dict = query_engine_dict or {}
        super().__init__(callback_manager)

    def _query_rec(
        self, query_bundle: QueryBundle, query_id: Optional[str] = None
    ) -> RESPONSE_TYPE:
        """Query recursively."""
        query_id = query_id or self._root_id
        query_engine = self._query_engine_dict[query_id]
        if isinstance(query_engine, RetrieverQueryEngine):
            retrieve_event_id = self.callback_manager.on_event_start(
                CBEventType.RETRIEVE
            )
            nodes = query_engine.retrieve(query_bundle)
            self.callback_manager.on_event_end(
                CBEventType.RETRIEVE,
                payload={EventPayload.NODES: nodes},
                event_id=retrieve_event_id,
            )
            nodes_for_synthesis = []
            for node_with_score in nodes:
                if isinstance(node_with_score.node, IndexNode):
                    sub_resp = self._query_rec(
                        query_bundle, query_id=node_with_score.node.index_id
                    )
                    node = TextNode(text=str(sub_resp))
                    node_to_add = NodeWithScore(node=node, score=1.0)
                else:
                    assert isinstance(node_with_score.node, TextNode)
                    node_to_add = node_with_score
                nodes_for_synthesis.append(node_to_add)
            response = query_engine.synthesize(query_bundle, nodes_for_synthesis, [])
            return response
        else:
            return query_engine.query(query_bundle)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query_rec(query_bundle, query_id=None)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query_rec(query_bundle, query_id=None)
