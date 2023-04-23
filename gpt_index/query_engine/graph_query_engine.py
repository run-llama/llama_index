from typing import Any, Dict, List, Optional, Tuple

from gpt_index.data_structs.node_v2 import IndexNode, Node, NodeWithScore
from gpt_index.indices.base_retriever import BaseRetriever
from gpt_index.indices.composability.graph import ComposableGraph
from gpt_index.indices.postprocessor.node import BaseNodePostprocessor
from gpt_index.indices.query.base import BaseQueryEngine
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.query.response_synthesis import ResponseSynthesizer
from gpt_index.indices.response.type import ResponseMode
from gpt_index.indices.service_context import ServiceContext
from gpt_index.optimization.optimizer import BaseTokenUsageOptimizer
from gpt_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    SimpleInputPrompt,
)
from gpt_index.response.schema import RESPONSE_TYPE


class ComposableGraphQueryEngine(BaseQueryEngine):
    def __init__(
        self,
        graph: ComposableGraph,
        response_synthesizer: Optional[ResponseSynthesizer] = None,
        custom_retrievers: Optional[Dict[str, BaseRetriever]] = None,
        recursive: bool = True,
    ) -> None:
        self._graph = graph
        self._response_synthesizer = (
            response_synthesizer or ResponseSynthesizer.from_args()
        )
        self._custom_retrievers = custom_retrievers or {}

        # additional configs
        self._recursive = recursive

    @classmethod
    def from_args(
        cls,
        graph: ComposableGraph,
        custom_retrievers: Optional[Dict[str, BaseRetriever]] = None,
        # response synthesizer args
        service_context: Optional[ServiceContext] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        simple_template: Optional[SimpleInputPrompt] = None,
        response_mode: ResponseMode = ResponseMode.DEFAULT,
        response_kwargs: Optional[Dict] = None,
        use_async: bool = False,
        streaming: bool = False,
        optimizer: Optional[BaseTokenUsageOptimizer] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        verbose: bool = False,
        # class-specific args
        **kwargs: Any,
    ) -> "ComposableGraphQueryEngine":
        response_synthesizer = ResponseSynthesizer.from_args(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            simple_template=simple_template,
            response_mode=response_mode,
            response_kwargs=response_kwargs,
            use_async=use_async,
            streaming=streaming,
            optimizer=optimizer,
            node_postprocessors=node_postprocessors,
            verbose=verbose,
        )
        return cls(
            graph=graph,
            custom_retrievers=custom_retrievers,
            response_synthesizer=response_synthesizer,
        )

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query_index(query_bundle, index_id=None, level=0)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query_index(query_bundle, index_id=None, level=0)

    def _query_index(
        self,
        query_bundle: QueryBundle,
        index_id: Optional[str] = None,
        level: int = 0,
    ) -> RESPONSE_TYPE:
        index_id = index_id or self._graph.root_id

        # get retriever
        if index_id in self._custom_retrievers:
            retriever = self._custom_retrievers[index_id]
        else:
            retriever = self._graph.get_index(index_id).as_retriever()
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
