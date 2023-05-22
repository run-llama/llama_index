from typing import Any, Dict, List, Optional, Sequence

from llama_index import Document
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType
from llama_index.data_structs.node import NodeWithScore
from llama_index.evaluation import QueryResponseEvaluator
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
from llama_index.response.schema import RESPONSE_TYPE, Response, StreamingResponse


class RetryQueryEngine(BaseQueryEngine):
    """Retriever query engine with retry.

    Args:
        retriever (BaseRetriever): A retriever object.
        response_synthesizer (Optional[ResponseSynthesizer]): A ResponseSynthesizer
            object.

    """

    def __init__(
        self,
        retriever: BaseRetriever,
        service_context: ServiceContext,
        response_synthesizer: Optional[ResponseSynthesizer] = None,
        callback_manager: Optional[CallbackManager] = None,
        check_source: bool = True,
        check_binary: bool = True,
        use_shrinking_percentile_cutoff: bool = True,
        max_retries: int = 3,
    ) -> None:
        self._retriever = retriever
        self._response_synthesizer = (
            response_synthesizer or ResponseSynthesizer.from_args()
        )
        self.callback_manager = callback_manager or CallbackManager([])
        self.service_context = service_context
        self.check_source = check_source
        self.check_binary = check_binary
        self.use_shrinking_percentile_cutoff = use_shrinking_percentile_cutoff
        self.max_retries = max_retries

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
        check_source: bool = True,
        check_binary: bool = True,
        use_shrinking_percentile_cutoff: bool = True,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> "RetryQueryEngine":
        """Initialize a RetryQueryEngine object."

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
            check_source (bool): Whether to check source nodes for relevance
                and discard irrelevant ones.
            check_binary (bool): Whether to check answer for correctness.
            use_shrinking_percentile_cutoff (bool): Whether to use shrinking percentiles
                cutoff on node text sentences on retry.
            max_retries (int): Maximum number of retries. Will shrink by 1 on each retry.

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

        service_context = service_context or ServiceContext.from_defaults()

        return cls(
            retriever=retriever,
            service_context=service_context,
            response_synthesizer=response_synthesizer,
            callback_manager=callback_manager,
            check_source=check_source,
            check_binary=check_binary,
            use_shrinking_percentile_cutoff=use_shrinking_percentile_cutoff,
            max_retries=max_retries,
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

        synth_id = self.callback_manager.on_event_start(CBEventType.SYNTHESIZE)
        response = self._response_synthesizer.synthesize(
            query_bundle=query_bundle,
            nodes=nodes,
        )
        self.callback_manager.on_event_end(
            CBEventType.SYNTHESIZE, payload={"response": response}, event_id=synth_id
        )

        self.callback_manager.on_event_end(CBEventType.QUERY, event_id=query_id)

        # Insert retry here
        # TODO: Add callbacks.
        if self.max_retries > 0:
            evaluator = QueryResponseEvaluator(service_context=self.service_context)

            if type(response) is StreamingResponse:
                typed_response = response.get_response()
            elif type(response) is Response:
                typed_response = response
            else:
                raise ValueError(f"Unexpected response type {type(response)}")

            if self.check_binary:
                response_eval = evaluator.evaluate(
                    query_bundle.query_str, typed_response
                )
                if response_eval == "YES":
                    return response
                # else continue
            if self.check_source:
                source_node_evals = evaluator.evaluate_source_nodes(
                    query_bundle.query_str, typed_response
                )
                old_source_nodes = evaluator.get_context(typed_response)
                assert len(old_source_nodes) == len(source_node_evals)
                new_source_docs: List[Document] = []
                for i in range(len(source_node_evals)):
                    if eval == "YES":
                        new_source_docs.append(old_source_nodes[i])
                if len(new_source_docs) == 0:
                    # TODO: Add context-less query.
                    raise ValueError("No source nodes are relevant")
        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """TODO: Answer a query."""
        query_id = self.callback_manager.on_event_start(CBEventType.QUERY)

        retrieve_id = self.callback_manager.on_event_start(CBEventType.RETRIEVE)
        nodes = self._retriever.retrieve(query_bundle)
        self.callback_manager.on_event_end(
            CBEventType.RETRIEVE, payload={"nodes": nodes}, event_id=retrieve_id
        )

        synth_id = self.callback_manager.on_event_start(CBEventType.SYNTHESIZE)
        response = await self._response_synthesizer.asynthesize(
            query_bundle=query_bundle,
            nodes=nodes,
        )
        self.callback_manager.on_event_end(
            CBEventType.SYNTHESIZE, payload={"response": response}, event_id=synth_id
        )

        self.callback_manager.on_event_end(CBEventType.QUERY, event_id=query_id)
        return response
