from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.indices.composability.graph import ComposableGraph
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
)
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


class ComposableGraphQueryEngine(BaseQueryEngine):
    """Composable graph query engine.

    This query engine can operate over a ComposableGraph.
    It can take in custom query engines for its sub-indices.

    Args:
        graph (ComposableGraph): A ComposableGraph object.
        custom_query_engines (Optional[Dict[str, BaseQueryEngine]]): A dictionary of
            custom query engines.
        recursive (bool): Whether to recursively query the graph.
        **kwargs: additional arguments to be passed to the underlying index query
            engine.

    """

    def __init__(
        self,
        graph: ComposableGraph,
        custom_query_engines: Optional[Dict[str, BaseQueryEngine]] = None,
        recursive: bool = True,
        **kwargs: Any
    ) -> None:
        """Init params."""
        self._graph = graph
        self._custom_query_engines = custom_query_engines or {}
        self._kwargs = kwargs

        # additional configs
        self._recursive = recursive
        callback_manager = callback_manager_from_settings_or_context(
            Settings, self._graph.service_context
        )
        super().__init__(callback_manager=callback_manager)

    def _get_prompt_modules(self) -> Dict[str, Any]:
        """Get prompt modules."""
        return {}

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query_index(query_bundle, index_id=None, level=0)

    @dispatcher.span
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query_index(query_bundle, index_id=None, level=0)

    def _query_index(
        self,
        query_bundle: QueryBundle,
        index_id: Optional[str] = None,
        level: int = 0,
    ) -> RESPONSE_TYPE:
        """Query a single index."""
        index_id = index_id or self._graph.root_id

        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            # get query engine
            if index_id in self._custom_query_engines:
                query_engine = self._custom_query_engines[index_id]
            else:
                query_engine = self._graph.get_index(index_id).as_query_engine(
                    **self._kwargs
                )

            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = query_engine.retrieve(query_bundle)
                retrieve_event.on_end(payload={EventPayload.NODES: nodes})

            if self._recursive:
                # do recursion here
                nodes_for_synthesis = []
                additional_source_nodes = []
                for node_with_score in nodes:
                    node_with_score, source_nodes = self._fetch_recursive_nodes(
                        node_with_score, query_bundle, level
                    )
                    nodes_for_synthesis.append(node_with_score)
                    additional_source_nodes.extend(source_nodes)
                response = query_engine.synthesize(
                    query_bundle, nodes_for_synthesis, additional_source_nodes
                )
            else:
                response = query_engine.synthesize(query_bundle, nodes)

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    def _fetch_recursive_nodes(
        self,
        node_with_score: NodeWithScore,
        query_bundle: QueryBundle,
        level: int,
    ) -> Tuple[NodeWithScore, List[NodeWithScore]]:
        """Fetch nodes.

        Uses existing node if it's not an index node.
        Otherwise fetch response from corresponding index.

        """
        if isinstance(node_with_score.node, IndexNode):
            index_node = node_with_score.node
            # recursive call
            response = self._query_index(query_bundle, index_node.index_id, level + 1)

            new_node = TextNode(text=str(response))
            new_node_with_score = NodeWithScore(
                node=new_node, score=node_with_score.score
            )
            return new_node_with_score, response.source_nodes
        else:
            return node_with_score, []
