from typing import Any, Dict, List, Optional, Tuple, Union
from gpt_index.data_structs.node_v2 import IndexNode, Node, NodeWithScore
from gpt_index.indices.common.base_retriever import BaseRetriever
from gpt_index.indices.postprocessor.node import BaseNodePostprocessor
from gpt_index.indices.query.base import BaseQueryEngine
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.response_synthesis import ResponseSynthesizer
from gpt_index.response.schema import RESPONSE_TYPE


class ComposableGraphQueryEngine(BaseQueryEngine):
    def __init__(
        self,
        graph: Any,
        response_synthesizer: Optional[ResponseSynthesizer] = None,
        retriever_id_kwargs: Optional[Dict[str, dict]] = None,
        retriever_type_kwargs: Optional[Dict[str, dict]] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        recursive: bool = True,
    ) -> None:
        from gpt_index.indices.composability.graph import ComposableGraph

        assert isinstance(graph, ComposableGraph)
        self._graph = graph
        self._response_synthesizer = (
            response_synthesizer or ResponseSynthesizer.from_args()
        )
        self._retriever_id_kwargs = retriever_id_kwargs or {}
        self._retriever_type_kwargs = retriever_type_kwargs or {}
        self._node_postprocessors = node_postprocessors

        # additional configs
        self._recursive = recursive

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query_index(query_bundle, index_id=None, level=0)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query_index(query_bundle, index_id=None, level=0)

    def _query_index(
        self,
        query_bundle: Union[str, QueryBundle],
        index_id: Optional[str] = None,
        level: int = 0,
    ) -> RESPONSE_TYPE:
        index_id = index_id or self._graph.root_id
        index_type = self._graph.get_index(index_id).index_struct.get_type()

        # get retriever args
        retriever_kwarg = {}
        retriever_kwarg.update(self._retriever_type_kwargs.get(index_type, {}))
        retriever_kwarg.update(self._retriever_id_kwargs.get(index_id, {}))

        retriever = self._graph.get_index(index_id).as_retriever(**retriever_kwarg)
        nodes = retriever.retrieve(query_bundle)

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
            response = self._response_synthesizer.synthesize(
                query_bundle, nodes_for_synthesis, additional_source_nodes
            )
        else:
            response = self._response_synthesizer.synthesize(query_bundle, nodes)

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

            new_node = Node(text=str(response))
            new_node_with_score = NodeWithScore(
                node=new_node, score=node_with_score.score
            )
            return new_node_with_score, response.source_nodes
        else:
            return node_with_score, []
