from typing import Any, Dict, List, Optional, Sequence

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType
from llama_index.data_structs.node import NodeWithScore
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.response_synthesis import ResponseSynthesizer
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.response.type import ResponseMode
from llama_index.indices.service_context import ServiceContext
from llama_index.optimization.optimizer import BaseTokenUsageOptimizer
from llama_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    SimpleInputPrompt,
)
from llama_index.response.schema import RESPONSE_TYPE


class RetrieverQueryEngine(BaseQueryEngine):
    """Retriever query engine.

    Args:
        retriever (BaseRetriever): A retriever object.
        response_synthesizer (Optional[ResponseSynthesizer]): A ResponseSynthesizer
            object.
        callback_manager (Optional[CallbackManager]): A callback manager.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        response_synthesizer: Optional[ResponseSynthesizer] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        self._response_synthesizer = (
            response_synthesizer
            or ResponseSynthesizer.from_args(callback_manager=callback_manager)
        )
        super().__init__(callback_manager)

    @classmethod
    def from_args(
        cls,
        retriever: BaseRetriever,
        service_context: Optional[ServiceContext] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        verbose: bool = False,
        # response synthesizer args
        response_mode: ResponseMode = ResponseMode.COMPACT,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        simple_template: Optional[SimpleInputPrompt] = None,
        response_kwargs: Optional[Dict] = None,
        use_async: bool = False,
        streaming: bool = False,
        optimizer: Optional[BaseTokenUsageOptimizer] = None,
        # class-specific args
        **kwargs: Any,
    ) -> "RetrieverQueryEngine":
        """Initialize a RetrieverQueryEngine object."

        Args:
            retriever (BaseRetriever): A retriever object.
            service_context (Optional[ServiceContext]): A ServiceContext object.
            node_postprocessors (Optional[List[BaseNodePostprocessor]]): A list of
                node postprocessors.
            verbose (bool): Whether to print out debug info.
            response_mode (ResponseMode): A ResponseMode object.
            text_qa_template (Optional[QuestionAnswerPrompt]): A QuestionAnswerPrompt
                object.
            refine_template (Optional[RefinePrompt]): A RefinePrompt object.
            simple_template (Optional[SimpleInputPrompt]): A SimpleInputPrompt object.
            response_kwargs (Optional[Dict]): A dict of response kwargs.
            use_async (bool): Whether to use async.
            streaming (bool): Whether to use streaming.
            optimizer (Optional[BaseTokenUsageOptimizer]): A BaseTokenUsageOptimizer
                object.

        """
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

        callback_manager = (
            service_context.callback_manager if service_context else CallbackManager([])
        )

        return cls(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            callback_manager=callback_manager,
        )

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retriever.retrieve(query_bundle)

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
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
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        return await self._response_synthesizer.asynthesize(
            query_bundle=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        query_id = self.callback_manager.on_event_start(CBEventType.QUERY)

        retrieve_id = self.callback_manager.on_event_start(CBEventType.RETRIEVE)
        nodes = self._retriever.retrieve(query_bundle)
        self.callback_manager.on_event_end(
            CBEventType.RETRIEVE, payload={"nodes": nodes}, event_id=retrieve_id
        )

        response = self._response_synthesizer.synthesize(
            query_bundle=query_bundle,
            nodes=nodes,
        )

        self.callback_manager.on_event_end(CBEventType.QUERY, event_id=query_id)
        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        query_id = self.callback_manager.on_event_start(CBEventType.QUERY)

        retrieve_id = self.callback_manager.on_event_start(CBEventType.RETRIEVE)
        nodes = self._retriever.retrieve(query_bundle)
        self.callback_manager.on_event_end(
            CBEventType.RETRIEVE, payload={"nodes": nodes}, event_id=retrieve_id
        )

        response = await self._response_synthesizer.asynthesize(
            query_bundle=query_bundle,
            nodes=nodes,
        )

        self.callback_manager.on_event_end(CBEventType.QUERY, event_id=query_id)
        return response

    @property
    def retriever(self) -> BaseRetriever:
        """Get the retriever object."""
        return self._retriever
