import logging
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
)

from gpt_index.data_structs.node_v2 import Node, NodeWithScore
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.response_builder import (
    BaseResponseBuilder,
    ResponseMode,
    get_response_builder,
)
from gpt_index.indices.service_context import ServiceContext
from gpt_index.optimization.optimizer import BaseTokenUsageOptimizer
from gpt_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    SimpleInputPrompt,
)
from gpt_index.response.schema import (
    RESPONSE_TYPE,
    Response,
    StreamingResponse,
)
from gpt_index.types import RESPONSE_TEXT_TYPE

logger = logging.getLogger(__name__)


class ResponseSynthesizer:
    def __init__(
        self,
        response_builder: Optional[BaseResponseBuilder],
        response_mode: ResponseMode,
        response_kwargs: Optional[Dict] = None,
        optimizer: Optional[BaseTokenUsageOptimizer] = None,
        verbose: bool = False,
    ) -> None:
        self._response_builder = response_builder
        self._response_mode = response_mode
        self._response_kwargs = response_kwargs or {}
        self._optimizer = optimizer
        self._verbose = verbose

    @classmethod
    def from_args(
        cls,
        service_context: ServiceContext,
        streaming: bool = False,
        use_async: bool = False,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        simple_template: Optional[SimpleInputPrompt] = None,
        response_mode: ResponseMode = ResponseMode.DEFAULT,
        response_kwargs: Optional[Dict] = None,
        optimizer: Optional[BaseTokenUsageOptimizer] = None,
    ) -> "ResponseSynthesizer":
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
        return cls(response_builder, response_mode, response_kwargs, optimizer)

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
            raise ValueError("Response must be a string or a generator.")

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: Sequence[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
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

        return self._prepare_response_output(response_str, source_nodes)

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: Sequence[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
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

        return self._prepare_response_output(response_str, source_nodes)
