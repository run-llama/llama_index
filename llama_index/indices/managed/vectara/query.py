from typing import Any, List, Optional

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.core import BaseQueryEngine, BaseRetriever
from llama_index.indices.managed.vectara.retriever import VectaraRetriever
from llama_index.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.schema import NodeWithScore, QueryBundle


class VectaraQueryEngine(BaseQueryEngine):
    """Retriever query engine for Vectara.

    Args:
        retriever (VectaraRetriever): A retriever object.
        summary_kwargs (dict): Additional kwargs to pass to the Vectara summary synthesizer.
    """

    def __init__(
        self,
        retriever: VectaraRetriever,
        summary_enabled: bool = False,
        summary_kwargs: Optional[dict] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        self._summary_enabled = summary_enabled
        if summary_enabled:
            self._summary_kwargs = {
                "summary_response_lang": summary_kwargs.get(
                    "summary_response_lang", "en"
                ),
                "summary_num_results": summary_kwargs.get("summary_num_results", 7),
                "summary_prompt_name": summary_kwargs.get(
                    "summary_prompt_name", "vectara-summary-ext-v1.2.0"
                ),
            }
        super().__init__(callback_manager=callback_manager)

    @classmethod
    def from_args(
        cls,
        retriever: VectaraRetriever,
        summary_enabled: bool = False,
        summary_kwargs: dict = {},
        **kwargs: Any,
    ) -> "VectaraQueryEngine":
        """Initialize a VectaraQueryEngine object.".

        Args:
            retriever (VectaraRetriever): A Vectara retriever object.
            summary_kwargs: additional keywords to pass to the Vectara summary synthesizer.

        """
        return cls(
            retriever=retriever,
            summary_enabled=summary_enabled,
            summary_kwargs=summary_kwargs,
        )

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retriever.retrieve(query_bundle)

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return await self._retriever.aretrieve(query_bundle)

    def with_retriever(self, retriever: VectaraRetriever) -> "VectaraQueryEngine":
        return VectaraQueryEngine(
            retriever=retriever,
            summary_enabled=self._summary_enabled,
            summary_kwargs=self._summary_kwargs,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes, response = self._retriever._vectara_query(
                query_bundle,
                kwargs=self._summary_kwargs if self._summary_enabled else {},
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
