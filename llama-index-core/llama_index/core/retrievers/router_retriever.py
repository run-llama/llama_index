"""Router retriever."""

import asyncio
import logging
from typing import List, Optional, Sequence

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.base_selector import BaseSelector
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle
from llama_index.core.selectors.utils import get_selector_from_llm
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)
from llama_index.core.tools.retriever_tool import RetrieverTool

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

    """

    def __init__(
        self,
        selector: BaseSelector,
        retriever_tools: Sequence[RetrieverTool],
        llm: Optional[LLM] = None,
        service_context: Optional[ServiceContext] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
    ) -> None:
        self._llm = llm or llm_from_settings_or_context(Settings, service_context)
        self._selector = selector
        self._retrievers: List[BaseRetriever] = [x.retriever for x in retriever_tools]
        self._metadatas = [x.metadata for x in retriever_tools]

        super().__init__(
            callback_manager=callback_manager_from_settings_or_context(
                Settings, service_context
            ),
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        # NOTE: don't include tools for now
        return {"selector": self._selector}

    @classmethod
    def from_defaults(
        cls,
        retriever_tools: Sequence[RetrieverTool],
        llm: Optional[LLM] = None,
        service_context: Optional[ServiceContext] = None,
        selector: Optional[BaseSelector] = None,
        select_multi: bool = False,
    ) -> "RouterRetriever":
        llm = llm or llm_from_settings_or_context(Settings, service_context)
        selector = selector or get_selector_from_llm(llm, is_multi=select_multi)

        return cls(
            selector,
            retriever_tools,
            llm=llm,
            service_context=service_context,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        with self.callback_manager.event(
            CBEventType.RETRIEVE,
            payload={EventPayload.QUERY_STR: query_bundle.query_str},
        ) as query_event:
            result = self._selector.select(self._metadatas, query_bundle)

            if len(result.inds) > 1:
                retrieved_results = {}
                for i, engine_ind in enumerate(result.inds):
                    logger.info(
                        f"Selecting retriever {engine_ind}: " f"{result.reasons[i]}."
                    )
                    selected_retriever = self._retrievers[engine_ind]
                    cur_results = selected_retriever.retrieve(query_bundle)
                    retrieved_results.update({n.node.node_id: n for n in cur_results})
            else:
                try:
                    selected_retriever = self._retrievers[result.ind]
                    logger.info(f"Selecting retriever {result.ind}: {result.reason}.")
                except ValueError as e:
                    raise ValueError("Failed to select retriever") from e

                cur_results = selected_retriever.retrieve(query_bundle)
                retrieved_results = {n.node.node_id: n for n in cur_results}

            query_event.on_end(payload={EventPayload.NODES: retrieved_results.values()})

        return list(retrieved_results.values())

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        with self.callback_manager.event(
            CBEventType.RETRIEVE,
            payload={EventPayload.QUERY_STR: query_bundle.query_str},
        ) as query_event:
            result = await self._selector.aselect(self._metadatas, query_bundle)

            if len(result.inds) > 1:
                retrieved_results = {}
                tasks = []
                for i, engine_ind in enumerate(result.inds):
                    logger.info(
                        f"Selecting retriever {engine_ind}: " f"{result.reasons[i]}."
                    )
                    selected_retriever = self._retrievers[engine_ind]
                    tasks.append(selected_retriever.aretrieve(query_bundle))

                results_of_results = await asyncio.gather(*tasks)
                cur_results = [
                    item for sublist in results_of_results for item in sublist
                ]
                retrieved_results.update({n.node.node_id: n for n in cur_results})
            else:
                try:
                    selected_retriever = self._retrievers[result.ind]
                    logger.info(f"Selecting retriever {result.ind}: {result.reason}.")
                except ValueError as e:
                    raise ValueError("Failed to select retriever") from e

                cur_results = await selected_retriever.aretrieve(query_bundle)
                retrieved_results = {n.node.node_id: n for n in cur_results}

            query_event.on_end(payload={EventPayload.NODES: retrieved_results.values()})

        return list(retrieved_results.values())
