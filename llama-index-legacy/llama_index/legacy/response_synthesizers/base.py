"""Response builder class.

This class provides general functions for taking in a set of text
and generating a response.

Will support different modes, from 1) stuffing chunks into prompt,
2) create and refine separately over each chunk, 3) tree summarization.

"""

import logging
from abc import abstractmethod
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

from llama_index.legacy.bridge.pydantic import BaseModel, Field
from llama_index.legacy.callbacks.base import CallbackManager
from llama_index.legacy.callbacks.schema import CBEventType, EventPayload
from llama_index.legacy.core.query_pipeline.query_component import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from llama_index.legacy.core.response.schema import (
    RESPONSE_TYPE,
    PydanticResponse,
    Response,
    StreamingResponse,
)
from llama_index.legacy.prompts.mixin import PromptMixin
from llama_index.legacy.schema import BaseNode, MetadataMode, NodeWithScore, QueryBundle
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.types import RESPONSE_TEXT_TYPE

logger = logging.getLogger(__name__)

QueryTextType = Union[str, QueryBundle]


class BaseSynthesizer(ChainableMixin, PromptMixin):
    """Response builder class."""

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        streaming: bool = False,
        output_cls: BaseModel = None,
    ) -> None:
        """Init params."""
        self._service_context = service_context or ServiceContext.from_defaults()
        self._callback_manager = self._service_context.callback_manager
        self._streaming = streaming
        self._output_cls = output_cls

    def _get_prompt_modules(self) -> Dict[str, Any]:
        """Get prompt modules."""
        # TODO: keep this for now since response synthesizers don't generally have sub-modules
        return {}

    @property
    def service_context(self) -> ServiceContext:
        return self._service_context

    @property
    def callback_manager(self) -> CallbackManager:
        return self._callback_manager

    @callback_manager.setter
    def callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        self._callback_manager = callback_manager
        # TODO: please fix this later
        self._service_context.callback_manager = callback_manager
        self._service_context.llm.callback_manager = callback_manager
        self._service_context.embed_model.callback_manager = callback_manager
        self._service_context.node_parser.callback_manager = callback_manager

    @abstractmethod
    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        ...

    @abstractmethod
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        ...

    def _log_prompt_and_response(
        self,
        formatted_prompt: str,
        response: RESPONSE_TEXT_TYPE,
        log_prefix: str = "",
    ) -> None:
        """Log prompt and response from LLM."""
        logger.debug(f"> {log_prefix} prompt template: {formatted_prompt}")
        self._service_context.llama_logger.add_log(
            {"formatted_prompt_template": formatted_prompt}
        )
        logger.debug(f"> {log_prefix} response: {response}")
        self._service_context.llama_logger.add_log(
            {f"{log_prefix.lower()}_response": response or "Empty Response"}
        )

    def _get_metadata_for_response(
        self,
        nodes: List[BaseNode],
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for response."""
        return {node.node_id: node.metadata for node in nodes}

    def _prepare_response_output(
        self,
        response_str: Optional[RESPONSE_TEXT_TYPE],
        source_nodes: List[NodeWithScore],
    ) -> RESPONSE_TYPE:
        """Prepare response object from response string."""
        response_metadata = self._get_metadata_for_response(
            [node_with_score.node for node_with_score in source_nodes]
        )

        if isinstance(response_str, str):
            return Response(
                response_str,
                source_nodes=source_nodes,
                metadata=response_metadata,
            )
        if isinstance(response_str, Generator):
            return StreamingResponse(
                response_str,
                source_nodes=source_nodes,
                metadata=response_metadata,
            )
        if isinstance(response_str, self._output_cls):
            return PydanticResponse(
                response_str, source_nodes=source_nodes, metadata=response_metadata
            )

        raise ValueError(
            f"Response must be a string or a generator. Found {type(response_str)}"
        )

    def synthesize(
        self,
        query: QueryTextType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        if len(nodes) == 0:
            return Response("Empty Response")

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE, payload={EventPayload.QUERY_STR: query.query_str}
        ) as event:
            response_str = self.get_response(
                query_str=query.query_str,
                text_chunks=[
                    n.node.get_content(metadata_mode=MetadataMode.LLM) for n in nodes
                ],
                **response_kwargs,
            )

            additional_source_nodes = additional_source_nodes or []
            source_nodes = list(nodes) + list(additional_source_nodes)

            response = self._prepare_response_output(response_str, source_nodes)

            event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def asynthesize(
        self,
        query: QueryTextType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        if len(nodes) == 0:
            return Response("Empty Response")

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE, payload={EventPayload.QUERY_STR: query.query_str}
        ) as event:
            response_str = await self.aget_response(
                query_str=query.query_str,
                text_chunks=[
                    n.node.get_content(metadata_mode=MetadataMode.LLM) for n in nodes
                ],
                **response_kwargs,
            )

            additional_source_nodes = additional_source_nodes or []
            source_nodes = list(nodes) + list(additional_source_nodes)

            response = self._prepare_response_output(response_str, source_nodes)

            event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """As query component."""
        return SynthesizerComponent(synthesizer=self)


class SynthesizerComponent(QueryComponent):
    """Synthesizer component."""

    synthesizer: BaseSynthesizer = Field(..., description="Synthesizer")

    class Config:
        arbitrary_types_allowed = True

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        self.synthesizer.callback_manager = callback_manager

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        # make sure both query_str and nodes are there
        if "query_str" not in input:
            raise ValueError("Input must have key 'query_str'")
        input["query_str"] = validate_and_convert_stringable(input["query_str"])

        if "nodes" not in input:
            raise ValueError("Input must have key 'nodes'")
        nodes = input["nodes"]
        if not isinstance(nodes, list):
            raise ValueError("Input nodes must be a list")
        for node in nodes:
            if not isinstance(node, NodeWithScore):
                raise ValueError("Input nodes must be a list of NodeWithScore")
        return input

    def _run_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run component."""
        output = self.synthesizer.synthesize(kwargs["query_str"], kwargs["nodes"])
        return {"output": output}

    async def _arun_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run component."""
        output = await self.synthesizer.asynthesize(
            kwargs["query_str"], kwargs["nodes"]
        )
        return {"output": output}

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"query_str", "nodes"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})
