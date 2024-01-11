from typing import Any, List, Optional

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.core.base_retriever import BaseRetriever
from llama_index.core.response.schema import RESPONSE_TYPE, Response
from llama_index.indices.managed.vectara.retriever import VectaraRetriever
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.schema import NodeWithScore, QueryBundle


class VectaraQueryEngine(BaseQueryEngine):
    """Retriever query engine for Vectara.

    Args:
        retriever (VectaraRetriever): A retriever object.
        summary_response_lang: response language for summary (ISO 639-2 code)
        summary_num_results: number of results to use for summary generation.
        summary_prompt_name: name of the prompt to use for summary generation.
    """

    def __init__(
        self,
        retriever: VectaraRetriever,
        summary_enabled: bool = False,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        summary_response_lang: str = "eng",
        summary_num_results: int = 5,
        summary_prompt_name: str = "vectara-experimental-summary-ext-2023-10-23-small",
    ) -> None:
        self._retriever = retriever
        self._summary_enabled = summary_enabled
        self._summary_response_lang = summary_response_lang
        self._summary_num_results = summary_num_results
        self._summary_prompt_name = summary_prompt_name
        self._node_postprocessors = node_postprocessors or []
        super().__init__(callback_manager=callback_manager)

    @classmethod
    def from_args(
        cls,
        retriever: VectaraRetriever,
        summary_enabled: bool = False,
        summary_response_lang: str = "eng",
        summary_num_results: int = 5,
        summary_prompt_name: str = "vectara-experimental-summary-ext-2023-10-23-small",
        **kwargs: Any,
    ) -> "VectaraQueryEngine":
        """Initialize a VectaraQueryEngine object.".

        Args:
            retriever (VectaraRetriever): A Vectara retriever object.
            summary_response_lang: response language for summary (ISO 639-2 code)
            summary_num_results: number of results to use for summary generation.
            summary_prompt_name: name of the prompt to use for summary generation.

        """
        return cls(
            retriever=retriever,
            summary_enabled=summary_enabled,
            summary_response_lang=summary_response_lang,
            summary_num_results=summary_num_results,
            summary_prompt_name=summary_prompt_name,
        )

    def _apply_node_postprocessors(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        for node_postprocessor in self._node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        return nodes

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self._retriever.aretrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    def with_retriever(self, retriever: VectaraRetriever) -> "VectaraQueryEngine":
        return VectaraQueryEngine(
            retriever=retriever,
            summary_enabled=self._summary_enabled,
            summary_response_lang=self._summary_response_lang,
            summary_num_results=self._summary_num_results,
            summary_prompt_name=self._summary_prompt_name,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes, response = self._retriever._vectara_query(
                query_bundle,
                kwargs={
                    "summary_response_lang": self._summary_response_lang,
                    "summary_num_results": self._summary_num_results,
                    "summary_prompt_name": self._summary_prompt_name,
                }
                if self._summary_enabled
                else {},
            )
            query_event.on_end(payload={EventPayload.RESPONSE: response})
        return Response(response=response, source_nodes=nodes)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query(query_bundle)

    @property
    def retriever(self) -> BaseRetriever:
        """Get the retriever object."""
        return self._retriever

    # required for PromptMixin
    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
