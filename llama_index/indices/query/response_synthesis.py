import logging
from typing import Any, Dict, Generator, List, Optional, Sequence

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType
from llama_index.data_structs.node import Node, NodeWithScore
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.response import (
    BaseResponseBuilder,
    ResponseMode,
    get_response_builder,
)
from llama_index.indices.service_context import ServiceContext
from llama_index.optimization.optimizer import BaseTokenUsageOptimizer
from llama_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    SimpleInputPrompt,
)
from llama_index.response.schema import RESPONSE_TYPE, Response, StreamingResponse
from llama_index.types import RESPONSE_TEXT_TYPE

logger = logging.getLogger(__name__)


class ResponseSynthesizer:
    """Response synthesize class.

    This class is responsible for synthesizing a response given a list of nodes.
    The way in which the response is synthesized depends on the response mode.

    Args:
        response_builder (Optional[BaseResponseBuilder]): A response builder object.
        response_mode (ResponseMode): A response mode.
        response_kwargs (Optional[Dict]): A dictionary of response kwargs.
        optimizer (Optional[BaseTokenUsageOptimizer]): A token usage optimizer.
        node_postprocessors (Optional[List[BaseNodePostprocessor]]): A list of node
            postprocessors.
        callback_manager (Optional[CallbackManager]): A callback manager.
        verbose (bool): Whether to print debug statements.

    """

    def __init__(
        self,
        response_builder: Optional[BaseResponseBuilder],
        response_mode: ResponseMode,
        response_kwargs: Optional[Dict] = None,
        optimizer: Optional[BaseTokenUsageOptimizer] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        self._response_builder = response_builder
        self._response_mode = response_mode
        self._response_kwargs = response_kwargs or {}
        self._optimizer = optimizer
        self._node_postprocessors = node_postprocessors or []
        self._callback_manager = callback_manager or CallbackManager([])
        self._verbose = verbose

    @classmethod
    def from_args(
        cls,
        service_context: Optional[ServiceContext] = None,
        streaming: bool = False,
        use_async: bool = False,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        simple_template: Optional[SimpleInputPrompt] = None,
        response_mode: ResponseMode = ResponseMode.COMPACT,
        response_kwargs: Optional[Dict] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        optimizer: Optional[BaseTokenUsageOptimizer] = None,
        verbose: bool = False,
    ) -> "ResponseSynthesizer":
        """Initialize response synthesizer from args.

        Args:
            service_context (Optional[ServiceContext]): A service context.
            streaming (bool): Whether to stream the response.
            use_async (bool): Whether to use async.
            text_qa_template (Optional[QuestionAnswerPrompt]): A text QA template.
            refine_template (Optional[RefinePrompt]): A refine template.
            simple_template (Optional[SimpleInputPrompt]): A simple template.
            response_mode (ResponseMode): A response mode.
            response_kwargs (Optional[Dict]): A dictionary of response kwargs.
            node_postprocessors (Optional[List[BaseNodePostprocessor]]): A list of node
                postprocessors.
            callback_manager (Optional[CallbackManager]): A callback manager.
            optimizer (Optional[BaseTokenUsageOptimizer]): A token usage optimizer.
            verbose (bool): Whether to print debug statements.

        """
        service_context = service_context or ServiceContext.from_defaults(
            callback_manager=callback_manager
        )

        # fallback to callback manager from service context
        callback_manager = callback_manager or service_context.callback_manager

        response_builder: Optional[BaseResponseBuilder] = None
        if response_mode != ResponseMode.NO_TEXT:
            response_builder = get_response_builder(
                service_context,
                text_qa_template,
                refine_template,
                simple_template,
                response_mode,
                use_async=use_async,
                streaming=streaming,
            )
        return cls(
            response_builder,
            response_mode,
            response_kwargs,
            optimizer,
            node_postprocessors,
            callback_manager,
            verbose,
        )

    def _get_extra_info_for_response(
        self,
        nodes: List[Node],
    ) -> Optional[Dict[str, Any]]:
        """Get extra info for response."""
        return {node.get_doc_id(): node.extra_info for node in nodes}

    def _prepare_response_output(
        self,
        response_str: Optional[RESPONSE_TEXT_TYPE],
        source_nodes: List[NodeWithScore],
    ) -> RESPONSE_TYPE:
        """Prepare response object from response string."""
        response_extra_info = self._get_extra_info_for_response(
            [node_with_score.node for node_with_score in source_nodes]
        )

        if response_str is None or isinstance(response_str, str):
            return Response(
                response_str,
                source_nodes=source_nodes,
                extra_info=response_extra_info,
            )
        elif response_str is None or isinstance(response_str, Generator):
            return StreamingResponse(
                response_str,
                source_nodes=source_nodes,
                extra_info=response_extra_info,
            )
        else:
            raise ValueError(
                f"Response must be a string or a generator. Found {type(response_str)}"
            )

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        event_id = self._callback_manager.on_event_start(CBEventType.SYNTHESIZE)
        for node_processor in self._node_postprocessors:
            nodes = node_processor.postprocess_nodes(nodes, query_bundle)

        text_chunks = []
        for node_with_score in nodes:
            text = node_with_score.node.get_text()
            if self._optimizer is not None:
                text = self._optimizer.optimize(query_bundle, text)
            text_chunks.append(text)

        if self._response_mode != ResponseMode.NO_TEXT:
            assert self._response_builder is not None
            response_str = self._response_builder.get_response(
                query_str=query_bundle.query_str,
                text_chunks=text_chunks,
                **self._response_kwargs,
            )
        else:
            response_str = None

        additional_source_nodes = additional_source_nodes or []
        source_nodes = list(nodes) + list(additional_source_nodes)

        response = self._prepare_response_output(response_str, source_nodes)
        self._callback_manager.on_event_end(
            CBEventType.SYNTHESIZE,
            payload={"response": response},
            event_id=event_id,
        )

        return response

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        event_id = self._callback_manager.on_event_start(CBEventType.SYNTHESIZE)
        for node_processor in self._node_postprocessors:
            nodes = node_processor.postprocess_nodes(nodes, query_bundle)

        text_chunks = []
        for node_with_score in nodes:
            text = node_with_score.node.get_text()
            if self._optimizer is not None:
                text = self._optimizer.optimize(query_bundle, text)
            text_chunks.append(text)

        if self._response_mode != ResponseMode.NO_TEXT:
            assert self._response_builder is not None
            response_str = await self._response_builder.aget_response(
                query_str=query_bundle.query_str,
                text_chunks=text_chunks,
                **self._response_kwargs,
            )
        else:
            response_str = None

        additional_source_nodes = additional_source_nodes or []
        source_nodes = list(nodes) + list(additional_source_nodes)

        response = self._prepare_response_output(response_str, source_nodes)
        self._callback_manager.on_event_end(
            CBEventType.SYNTHESIZE,
            payload={"response": response},
            event_id=event_id,
        )

        return response
