from typing import Dict, List, Optional, Tuple, Union
from gpt_index.data_structs.node_v2 import IndexNode, Node, NodeWithScore
from gpt_index.indices.common.base_retriever import BaseRetriever
from gpt_index.indices.composability.graph import ComposableGraph
from gpt_index.indices.postprocessor.node import BaseNodePostprocessor
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.response_synthesis import ResponseSynthesizer
from gpt_index.response.schema import RESPONSE_TYPE


class ComposedGraphQueryEngine:
    def __init__(
        self,
        graph: ComposableGraph,
        retrievers: Dict[str, BaseRetriever],
        response_synthesizer: ResponseSynthesizer,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        recursive: bool = True,
    ) -> None:
        self._graph = graph
        self._retrievers = retrievers
        self._response_synthesizer = response_synthesizer
        self._node_postprocessors = node_postprocessors

        # additional configs
        self._recursive = recursive

    def query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query(query_bundle, index_id=None, level=0)

    def _query(
        self,
        query_bundle: Union[str, QueryBundle],
        index_id: Optional[str] = None,
        level: int = 0,
    ) -> RESPONSE_TYPE:
        index_id = index_id or self._graph.root_id
        retriever = self._retrievers[index_id]
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
            response = self.query(query_bundle, index_node.index_id, level + 1)

            new_node = Node(text=str(response))
            new_node_with_score = NodeWithScore(
                node=new_node, score=node_with_score.score
            )
            return new_node_with_score, response.source_nodes
        else:
            return node_with_score, []
