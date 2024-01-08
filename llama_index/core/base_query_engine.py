"""Base query engine."""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from llama_index.bridge.pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.core.query_pipeline.query_component import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from llama_index.core.response.schema import RESPONSE_TYPE
from llama_index.prompts.mixin import PromptDictType, PromptMixin
from llama_index.schema import NodeWithScore, QueryBundle, QueryType

logger = logging.getLogger(__name__)


class BaseQueryEngine(ChainableMixin, PromptMixin):
    """Base query engine."""

    def __init__(self, callback_manager: Optional[CallbackManager]) -> None:
        self.callback_manager = callback_manager or CallbackManager([])

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            return self._query(str_or_query_bundle)

    async def aquery(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            return await self._aquery(str_or_query_bundle)

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

    @abstractmethod
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass

    @abstractmethod
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """Return a query component."""
        return QueryEngineComponent(query_engine=self)


class QueryEngineComponent(QueryComponent):
    """Query engine component."""

    query_engine: BaseQueryEngine = Field(..., description="Query engine")

    class Config:
        arbitrary_types_allowed = True

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
