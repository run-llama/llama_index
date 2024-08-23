"""Base query engine."""

import logging
from abc import ABCMeta
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.base.query_pipeline.query import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.bridge.pydantic import Field, ConfigDict, SerializeAsAny
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.prompts.mixin import PromptDictType, PromptMixin
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType
from llama_index.core.instrumentation.events.query import (
    QueryEndEvent,
    QueryStartEvent,
)
import llama_index.core.instrumentation as instrument
from llama_index.core.workflow.workflow import Workflow, _WorkflowMeta
from llama_index.core.async_utils import asyncio_run
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.bridge.pydantic import BaseModel

dispatcher = instrument.get_dispatcher(__name__)
logger = logging.getLogger(__name__)


PydanticMetaclass = type(BaseModel)


class CombinedMeta(_WorkflowMeta, ABCMeta):
    pass


class BaseQueryEngine(ChainableMixin, PromptMixin, Workflow, metaclass=CombinedMeta):
    """Base query engine."""

    def __init__(
        self,
        callback_manager: Optional[CallbackManager],
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        self.callback_manager = callback_manager or CallbackManager([])
        Workflow.__init__(self, timeout=timeout, **kwargs)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    @dispatcher.span
    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        dispatcher.event(QueryStartEvent(query=str_or_query_bundle))
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            query_result = self._query(str_or_query_bundle)
        dispatcher.event(
            QueryEndEvent(query=str_or_query_bundle, response=query_result)
        )
        return query_result

    @dispatcher.span
    async def aquery(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        dispatcher.event(QueryStartEvent(query=str_or_query_bundle))
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            query_result = await self._aquery(str_or_query_bundle)
        dispatcher.event(
            QueryEndEvent(query=str_or_query_bundle, response=query_result)
        )
        return query_result

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        raise NotImplementedError(
            "This query engine does not support retrieve, use query directly"
        )

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        raise NotImplementedError(
            "This query engine does not support synthesize, use query directly"
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        raise NotImplementedError(
            "This query engine does not support asynthesize, use aquery directly"
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return asyncio_run(self._aquery(query_bundle=query_bundle))

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            response = await self.run(query_bundle=query_bundle)
            query_event.on_end(payload={EventPayload.RESPONSE: response})
        return response

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """Return a query component."""
        return QueryEngineComponent(query_engine=self)


class QueryEngineComponent(QueryComponent):
    """Query engine component."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    query_engine: SerializeAsAny[BaseQueryEngine] = Field(
        ..., description="Query engine"
    )

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        self.query_engine.callback_manager = callback_manager

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        # make sure input is a string
        input["input"] = validate_and_convert_stringable(input["input"])
        return input

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        output = self.query_engine.query(kwargs["input"])
        return {"output": output}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component."""
        output = await self.query_engine.aquery(kwargs["input"])
        return {"output": output}

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"input"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})
