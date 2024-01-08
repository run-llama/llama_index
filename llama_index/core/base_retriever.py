"""Base retriever."""
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from llama_index.bridge.pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.core.query_pipeline.query_component import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from llama_index.prompts.mixin import PromptDictType, PromptMixin, PromptMixinType
from llama_index.schema import NodeWithScore, QueryBundle, QueryType
from llama_index.service_context import ServiceContext


class BaseRetriever(ChainableMixin, PromptMixin):
    """Base retriever."""

    def __init__(self, callback_manager: Optional[CallbackManager] = None) -> None:
        self.callback_manager = callback_manager or CallbackManager()

    def _check_callback_manager(self) -> None:
        """Check callback manager."""
        if not hasattr(self, "callback_manager"):
            self.callback_manager = CallbackManager()

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        """Retrieve nodes given query.

        Args:
            str_or_query_bundle (QueryType): Either a query string or
                a QueryBundle object.

        """
        self._check_callback_manager()

        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self._retrieve(query_bundle)
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        return nodes

    async def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        self._check_callback_manager()

        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self._aretrieve(query_bundle)
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        return nodes

    @abstractmethod
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query.

        Implemented by the user.

        """

    # TODO: make this abstract
    # @abstractmethod
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve nodes given query.

        Implemented by the user.

        """
        return self._retrieve(query_bundle)

    def get_service_context(self) -> Optional[ServiceContext]:
        """Attempts to resolve a service context.
        Short-circuits at self.service_context, self._service_context,
        or self._index.service_context.
        """
        if hasattr(self, "service_context"):
            return self.service_context
        if hasattr(self, "_service_context"):
            return self._service_context
        elif hasattr(self, "_index") and hasattr(self._index, "service_context"):
            return self._index.service_context
        return None

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """Return a query component."""
        return RetrieverComponent(retriever=self)


class RetrieverComponent(QueryComponent):
    """Retriever component."""

    retriever: BaseRetriever = Field(..., description="Retriever")

    class Config:
        arbitrary_types_allowed = True

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        self.retriever.callback_manager = callback_manager

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        # make sure input is a string
        input["input"] = validate_and_convert_stringable(input["input"])
        return input

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        output = self.retriever.retrieve(kwargs["input"])
        return {"output": output}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component."""
        output = await self.retriever.aretrieve(kwargs["input"])
        return {"output": output}

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"input"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})
