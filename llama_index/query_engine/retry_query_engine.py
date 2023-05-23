import logging
from typing import Any, Dict, List, Optional, Sequence

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType
from llama_index.data_structs.node import NodeWithScore
from llama_index.evaluation.base import QueryResponseEvaluator
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.list.base import GPTListIndex
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.response_synthesis import ResponseSynthesizer
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.response.type import ResponseMode
from llama_index.indices.service_context import ServiceContext
from llama_index.optimization.optimizer import (
    BaseTokenUsageOptimizer,
    SentenceEmbeddingOptimizer,
)
from llama_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    SimpleInputPrompt,
)
from llama_index.readers.schema.base import Document
from llama_index.response.schema import RESPONSE_TYPE, Response, StreamingResponse

logger = logging.getLogger(__name__)


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
        percentile_cutoff: float = 0.7,
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
        self.percentile_cutoff = percentile_cutoff

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
        percentile_cutoff: float = 0.7,
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
            percentile_cutoff (float): Percentile cutoff for sentence optimization
                on node text.
            use_shrinking_percentile_cutoff (bool): Whether to use shrinking percentiles
                cutoff on node text sentences on retry.
            max_retries (int): Maximum number of retries. Shrinks by 1 each time.

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
            percentile_cutoff=percentile_cutoff,
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
                else:
                    logger.debug(
                        f">Binary evaluator returned NO, response: \
                        {typed_response.response}"
                    )
                # else continue

            new_retriever = self._retriever
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
                    else:
                        logger.debug(
                            f">Source evaluator return NO, node: {old_source_nodes[i]}"
                        )
                if len(new_source_docs) == 0:
                    # TODO: Add context-less query.
                    logger.warn("No source nodes are relevant")
                if new_source_docs:
                    new_index = GPTListIndex.from_documents(new_source_docs)
                    new_retriever = new_index.as_retriever(
                        optimizer=SentenceEmbeddingOptimizer(
                            percentile_cutoff=self.percentile_cutoff
                        )
                    )

            if self.use_shrinking_percentile_cutoff:
                new_percentile_cutoff = self.percentile_cutoff - 0.1
            text_qa_template = QuestionAnswerPrompt(
                self.get_error_correcting_qa_tmpl(typed_response.response)
            )
            refine_template = RefinePrompt(
                self.get_error_correcting_refine_tmpl(typed_response.response)
            )
            new_query_engine = RetryQueryEngine.from_args(
                text_qa_template=text_qa_template,
                refine_template=refine_template,
                retriever=new_retriever,
                check_source=self.check_source,
                check_binary=self.check_binary,
                percentile_cutoff=new_percentile_cutoff,
                use_shrinking_percentile_cutoff=self.use_shrinking_percentile_cutoff,
                max_retries=self.max_retries - 1,
            )
            logger.debug(f">Retrying query. Retries left: {self.max_retries - 1}")
            response = new_query_engine.query(query_bundle)

        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Not supported."""
        return self._query(query_bundle)

    def get_error_correcting_qa_tmpl(self, neg_ex: Optional[str]) -> str:
        return (
            "Context information is below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "A previous bad answer is given below. \n"
            "---------------------\n"
            f"{neg_ex}"
            "\n---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the question: {query_str}\n"
        )

    def get_error_correcting_refine_tmpl(self, neg_ex: Optional[str]) -> str:
        return (
            "The original question is as follows: {query_str}\n"
            "We have provided an existing answer: {existing_answer}\n"
            f"We also have a previous bad answer: {neg_ex}\n"
            "We have the opportunity to refine the existing answer "
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{context_msg}\n"
            "------------\n"
            "Given the new context, refine the original answer to better "
            "answer the question. "
            "If the context isn't useful, return the original answer."
        )
