import logging
from typing import Callable, Optional, Sequence

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType
from llama_index.data_structs.node import Node
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.selectors.llm_selectors import LLMSingleSelector
from llama_index.selectors.types import BaseSelector
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.tools.types import ToolMetadata

logger = logging.getLogger(__name__)


class RouterQueryEngine(BaseQueryEngine):
    """Router query engine.

    Selects one out of several candidate query engines to execute a query.

    Args:
        selector (BaseSelector): A selector that chooses one out of many options based
            on each candidate's metadata and query.
        query_engine_tools (Sequence[QueryEngineTool]): A sequence of candidate
            query engines. They must be wrapped as tools to expose metadata to
            the selector.
        callback_manager (Optional[CallbackManager]): A callback manager.

    """

    def __init__(
        self,
        selector: BaseSelector,
        query_engine_tools: Sequence[QueryEngineTool],
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._selector = selector
        self._query_engines = [x.query_engine for x in query_engine_tools]
        self._metadatas = [x.metadata for x in query_engine_tools]

        if len(query_engine_tools) > 0:
            callback_manager = query_engine_tools[0].query_engine.callback_manager
        super().__init__(callback_manager)

    @classmethod
    def from_defaults(
        cls,
        query_engine_tools: Sequence[QueryEngineTool],
        selector: Optional[BaseSelector] = None,
    ) -> "RouterQueryEngine":
        selector = selector or LLMSingleSelector.from_defaults()
        return cls(selector, query_engine_tools)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        event_id = self.callback_manager.on_event_start(CBEventType.QUERY)
        result = self._selector.select(self._metadatas, query_bundle)
        try:
            selected_query_engine = self._query_engines[result.ind]
            logger.info(f"Selecting query engine {result.ind}: {result.reason}.")
        except ValueError as e:
            raise ValueError("Failed to select query engine") from e

        response = selected_query_engine.query(query_bundle)
        self.callback_manager.on_event_end(CBEventType.QUERY, event_id=event_id)
        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        result = await self._selector.aselect(self._metadatas, query_bundle)
        try:
            selected_query_engine = self._query_engines[result.ind]
            logger.info(f"Selecting query engine {result.ind}: {result.reason}.")
        except ValueError as e:
            raise ValueError("Failed to select query engine") from e

        response = await selected_query_engine.aquery(query_bundle)
        return response


def default_node_to_metadata_fn(node: Node) -> ToolMetadata:
    """Default node to metadata function.

    We use the node's text as the Tool description.

    """

    extra_info = node.extra_info or {}
    if "tool_name" not in extra_info:
        raise ValueError("Node must have a tool_name in extra_info.")
    return ToolMetadata(name=extra_info["tool_name"], description=node.get_text())


class RetrieverRouterQueryEngine(BaseQueryEngine):
    """Retriever-based router query engine.

    Use a retriever to select a set of Nodes. Each node will be converted
    into a ToolMetadata object, and also used to retrieve a query engine, to form
    a QueryEngineTool.

    NOTE: this is a beta feature. We are figuring out the right interface
    between the retriever and query engine.

    Args:
        selector (BaseSelector): A selector that chooses one out of many options based
            on each candidate's metadata and query.
        query_engine_tools (Sequence[QueryEngineTool]): A sequence of candidate
            query engines. They must be wrapped as tools to expose metadata to
            the selector.
        callback_manager (Optional[CallbackManager]): A callback manager.

    """

    def __init__(
        self,
        retriever: BaseRetriever,
        node_to_query_engine_fn: Callable,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        self._node_to_query_engine_fn = node_to_query_engine_fn
        super().__init__(callback_manager)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        nodes_with_score = self._retriever.retrieve(query_bundle)
        # TODO: for now we only support retrieving one node
        if len(nodes_with_score) > 1:
            raise ValueError("Retrieved more than one node.")

        node = nodes_with_score[0].node
        query_engine = self._node_to_query_engine_fn(node)
        return query_engine.query(query_bundle)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query(query_bundle)
