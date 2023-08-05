"""Router retriever."""

import logging
from typing import List, Optional, Sequence

from llama_index.schema import NodeWithScore
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.selectors.llm_selectors import LLMMultiSelector, LLMSingleSelector
from llama_index.selectors.types import BaseSelector
from llama_index.tools.retriever_tool import RetrieverTool

logger = logging.getLogger(__name__)


class RouterRetriever(BaseRetriever):
    """Router retriever.

    Selects one (or multiple) out of several candidate retrievers to execute a query.

    Args:
        selector (BaseSelector): A selector that chooses one out of many options based
            on each candidate's metadata and query.
        retriever_tools (Sequence[RetrieverTool]): A sequence of candidate
            retrievers. They must be wrapped as tools to expose metadata to
            the selector.
        service_context (Optional[ServiceContext]): A service context.

    """

    def __init__(
        self,
        selector: BaseSelector,
        retriever_tools: Sequence[RetrieverTool],
        service_context: Optional[ServiceContext] = None,
        combine_mode: str = "OR",
    ) -> None:
        self.service_context = service_context or ServiceContext.from_defaults()
        self._selector = selector
        self._retrievers: List[BaseRetriever] = [x.retriever for x in retriever_tools]
        self._metadatas = [x.metadata for x in retriever_tools]
        self.callback_manager = self.service_context.callback_manager
        self._combine_mode = combine_mode

    @classmethod
    def from_defaults(
        cls,
        retriever_tools: Sequence[RetrieverTool],
        service_context: Optional[ServiceContext] = None,
        selector: Optional[BaseSelector] = None,
        select_multi: bool = False,
    ) -> "RouterRetriever":
        if selector is None and select_multi:
            selector = LLMMultiSelector.from_defaults(service_context=service_context)
        elif selector is None and not select_multi:
            selector = LLMSingleSelector.from_defaults(service_context=service_context)

        assert selector is not None

        return cls(
            selector,
            retriever_tools,
            service_context=service_context,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        with self.callback_manager.event(
            CBEventType.RETRIEVE,
            payload={EventPayload.QUERY_STR: query_bundle.query_str},
        ) as query_event:
            result = self._selector.select(self._metadatas, query_bundle)

            if len(result.inds) > 1:
                results = []
                for i, engine_ind in enumerate(result.inds):
                    logger.info(
                        f"Selecting retriever {engine_ind}: " f"{result.reasons[i]}."
                    )
                    selected_retriever = self._retrievers[engine_ind]
                    results.extend(selected_retriever.retrieve(query_bundle))
            else:
                try:
                    selected_retriever = self._retrievers[result.ind]
                    logger.info(f"Selecting retriever {result.ind}: {result.reason}.")
                except ValueError as e:
                    raise ValueError("Failed to select retriever") from e

                final_response = selected_retriever.retrieve(query_bundle)

            query_event.on_end(payload={EventPayload.RESPONSE: final_response})

        return final_response

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        with self.callback_manager.event(
            CBEventType.RETRIEVE,
            payload={EventPayload.QUERY_STR: query_bundle.query_str},
        ) as query_event:
            result = await self._selector.aselect(self._metadatas, query_bundle)

            if len(result.inds) > 1:
                results = []
                for i, engine_ind in enumerate(result.inds):
                    logger.info(
                        f"Selecting retriever {engine_ind}: " f"{result.reasons[i]}."
                    )
                    selected_retriever = self._retrievers[engine_ind]
                    results.extend(await selected_retriever.aretrieve(query_bundle))
            else:
                try:
                    selected_retriever = self._retrievers[result.ind]
                    logger.info(f"Selecting retriever {result.ind}: {result.reason}.")
                except ValueError as e:
                    raise ValueError("Failed to select retriever") from e

                final_response = await selected_retriever.aretrieve(query_bundle)

            query_event.on_end(payload={EventPayload.RESPONSE: final_response})

        return final_response
