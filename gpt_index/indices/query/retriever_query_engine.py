from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence
from gpt_index.data_structs.node_v2 import NodeWithScore
from gpt_index.indices.common.base_retriever import BaseRetriever
from gpt_index.indices.postprocessor.node import BaseNodePostprocessor
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.response_synthesis import ResponseSynthesizer
from gpt_index.indices.response.type import ResponseMode
from gpt_index.indices.service_context import ServiceContext
from gpt_index.optimization.optimizer import BaseTokenUsageOptimizer
from gpt_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    SimpleInputPrompt,
)
from gpt_index.response.schema import RESPONSE_TYPE
from gpt_index.token_counter.token_counter import llm_token_counter


class BaseQueryEngine(ABC):
    @abstractmethod
    def query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass

    @abstractmethod
    async def aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass


class RetrieverQueryEngine:
    def __init__(
        self,
        retriever: BaseRetriever,
        response_synthesizer: ResponseSynthesizer,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    ) -> None:
        self._retriever = retriever
        self._response_synthesizer = response_synthesizer
        self._node_postprocessors = node_postprocessors

    @classmethod
    def from_args(
        cls,
        retriever: BaseRetriever,
        service_context: ServiceContext,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        verbose: bool = False,
        # response synthesizer args
        response_mode: ResponseMode = ResponseMode.DEFAULT,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        simple_template: Optional[SimpleInputPrompt] = None,
        response_kwargs: Optional[Dict] = None,
        use_async: bool = False,
        streaming: bool = False,
        optimizer: Optional[BaseTokenUsageOptimizer] = None,
        # class-specific args
        **kwargs: Any,
    ) -> "BaseQueryEngine":
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
        )
        return cls(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
            verbose=verbose,
            **kwargs,
        )

    @llm_token_counter("query")
    def query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        # TODO: support include summary
        nodes = self.retrieve(query_bundle)
        return self.synthesize(query_bundle, nodes)

    @llm_token_counter("query")
    async def aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        # TODO: support include summary
        nodes = self.retrieve(query_bundle)
        response = await self.asynthesize(query_bundle, nodes)
        return response

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Get list of tuples of node and similarity for response.

        First part of the tuple is the node.
        Second part of tuple is the distance from query to the node.
        If not applicable, it's None.
        """
        nodes = self._retriever.retrieve(query_bundle)

        postprocess_info = {
            "query_bundle": query_bundle,
        }
        for node_processor in self._node_postprocessors:
            nodes = node_processor.postprocess_nodes(nodes, postprocess_info)

        return nodes

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: Sequence[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        return self._response_synthesizer.synthesize(
            query_bundle=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: Sequence[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        return await self._response_synthesizer.asynthesize(
            query_bundle=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )
