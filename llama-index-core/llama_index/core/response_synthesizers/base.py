"""Response builder class.

This class provides general functions for taking in a set of text
and generating a response.

Will support different modes, from 1) stuffing chunks into prompt,
2) create and refine separately over each chunk, 3) tree summarization.

"""

import logging
from abc import abstractmethod
from typing import Any, Dict, Generator, List, Optional, Sequence, AsyncGenerator, Type

from llama_index.core.base.query_pipeline.query import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    PydanticResponse,
    Response,
    StreamingResponse,
    AsyncStreamingResponse,
)
from llama_index.core.bridge.pydantic import BaseModel, Field, ConfigDict
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms import LLM
from llama_index.core.prompts.mixin import PromptMixin
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeWithScore,
    QueryBundle,
    QueryType,
)
from llama_index.core.settings import Settings
from llama_index.core.types import RESPONSE_TEXT_TYPE
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.instrumentation.events.synthesis import (
    SynthesizeStartEvent,
    SynthesizeEndEvent,
)
from llama_index.core.llms.structured_llm import StructuredLLM
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)

logger = logging.getLogger(__name__)

QueryTextType = QueryType


def empty_response_generator() -> Generator[str, None, None]:
    yield "Empty Response"


async def empty_response_agenerator() -> AsyncGenerator[str, None]:
    yield "Empty Response"


class BaseSynthesizer(ChainableMixin, PromptMixin, DispatcherSpanMixin):
    """Response builder class."""

    def __init__(
        self,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        prompt_helper: Optional[PromptHelper] = None,
        streaming: bool = False,
        output_cls: Optional[Type[BaseModel]] = None,
    ) -> None:
        """Init params."""
        self._llm = llm or Settings.llm

        if callback_manager:
            self._llm.callback_manager = callback_manager

        self._callback_manager = callback_manager or Settings.callback_manager

        self._prompt_helper = (
            prompt_helper
            or Settings._prompt_helper
            or PromptHelper.from_llm_metadata(
                self._llm.metadata,
            )
        )

        self._streaming = streaming
        self._output_cls = output_cls

    def _get_prompt_modules(self) -> Dict[str, Any]:
        """Get prompt modules."""
        # TODO: keep this for now since response synthesizers don't generally have sub-modules
        return {}

    @property
    def callback_manager(self) -> CallbackManager:
        return self._callback_manager

    @callback_manager.setter
    def callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        self._callback_manager = callback_manager
        # TODO: please fix this later
        self._callback_manager = callback_manager
        self._llm.callback_manager = callback_manager

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
        logger.debug(f"> {log_prefix} response: {response}")

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

        if isinstance(self._llm, StructuredLLM):
            # convert string to output_cls
            output = self._llm.output_cls.model_validate_json(str(response_str))
            return PydanticResponse(
                output,
                source_nodes=source_nodes,
                metadata=response_metadata,
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
        if isinstance(response_str, AsyncGenerator):
            return AsyncStreamingResponse(
                response_str,
                source_nodes=source_nodes,
                metadata=response_metadata,
            )

        if self._output_cls is not None and isinstance(response_str, self._output_cls):
            return PydanticResponse(
                response_str, source_nodes=source_nodes, metadata=response_metadata
            )

        raise ValueError(
            f"Response must be a string or a generator. Found {type(response_str)}"
        )

    @dispatcher.span
    def synthesize(
        self,
        query: QueryTextType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        dispatcher.event(
            SynthesizeStartEvent(
                query=query,
            )
        )

        if len(nodes) == 0:
            if self._streaming:
                empty_response_stream = StreamingResponse(
                    response_gen=empty_response_generator()
                )
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response_stream,
                    )
                )
                return empty_response_stream
            else:
                empty_response = Response("Empty Response")
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response,
                    )
                )
                return empty_response

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE,
            payload={EventPayload.QUERY_STR: query.query_str},
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

        dispatcher.event(
            SynthesizeEndEvent(
                query=query,
                response=response,
            )
        )
        return response

    @dispatcher.span
    async def asynthesize(
        self,
        query: QueryTextType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        dispatcher.event(
            SynthesizeStartEvent(
                query=query,
            )
        )
        if len(nodes) == 0:
            if self._streaming:
                empty_response_stream = AsyncStreamingResponse(
                    response_gen=empty_response_agenerator()
                )
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response_stream,
                    )
                )
                return empty_response_stream
            else:
                empty_response = Response("Empty Response")
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response,
                    )
                )
                return empty_response

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE,
            payload={EventPayload.QUERY_STR: query.query_str},
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

        dispatcher.event(
            SynthesizeEndEvent(
                query=query,
                response=response,
            )
        )
        return response

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """As query component."""
        return SynthesizerComponent(synthesizer=self)


class SynthesizerComponent(QueryComponent):
    """Synthesizer component."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    synthesizer: BaseSynthesizer = Field(..., description="Synthesizer")

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
