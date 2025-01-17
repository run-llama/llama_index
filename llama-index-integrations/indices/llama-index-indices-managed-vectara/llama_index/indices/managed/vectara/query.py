from typing import Any, List, Dict, Optional

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.settings import (
    Settings,
)
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
)

from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    Response,
)
from llama_index.indices.managed.vectara.retriever import VectaraRetriever


class VectaraQueryEngine(BaseQueryEngine):
    """
    Retriever query engine for Vectara.

    Args:
        retriever (VectaraRetriever): A retriever object.
        streaming: whether to use streaming mode.
        summary_response_lang: response language for summary (ISO 639-2 code)
        summary_num_results: number of results to use for summary generation.
        summary_prompt_name: name of the prompt to use for summary generation.
    """

    def __init__(
        self,
        retriever: VectaraRetriever,
        streaming: bool = False,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        summary_enabled: bool = False,
        summary_response_lang: str = "eng",
        summary_num_results: int = 5,
        summary_prompt_name: str = "vectara-summary-ext-24-05-med-omni",
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self._retriever = retriever
        self._streaming = streaming
        self._summary_enabled = summary_enabled
        self._summary_response_lang = summary_response_lang
        self._summary_num_results = summary_num_results
        self._summary_prompt_name = summary_prompt_name
        self._node_postprocessors = node_postprocessors or []
        self._verbose = verbose
        super().__init__(callback_manager=callback_manager)

    @classmethod
    def from_args(
        cls,
        retriever: VectaraRetriever,
        streaming: bool = False,
        summary_enabled: bool = False,
        **kwargs: Any,
    ) -> "VectaraQueryEngine":
        """
        Initialize a VectaraQueryEngine object.".

        Args:
            retriever (VectaraRetriever): A Vectara retriever object.
            summary_enabled: is summary enabled

        """
        return cls(
            retriever=retriever,
            streaming=streaming,
            summary_enabled=summary_enabled,
            **kwargs,
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
            verbose=self._verbose,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        kwargs = (
            {
                "response_language": self._summary_response_lang,
                "max_used_search_results": self._summary_num_results,
                "generation_preset_name": self._summary_prompt_name,
            }
            if self._summary_enabled
            else {}
        )

        if self._streaming:
            query_response = self._retriever._vectara_stream(
                query_bundle, chat=False, verbose=self._verbose
            )
        else:
            nodes, response, _ = self._retriever._vectara_query(
                query_bundle, verbose=self._verbose, **kwargs
            )
            query_response = Response(
                response=response["text"],
                source_nodes=nodes,
                metadata={"fcs": response.get("fcs", None)},
            )

        return query_response

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


class VectaraChatEngine(BaseChatEngine):
    def __init__(
        self,
        retriever: VectaraRetriever,
        streaming: bool = False,
        summary_response_lang: str = "eng",
        summary_num_results: int = 5,
        summary_prompt_name: str = "vectara-summary-ext-24-05-med-omni",
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self._retriever = retriever
        self._streaming = streaming
        self._summary_enabled = True
        self._summary_response_lang = summary_response_lang
        self._summary_num_results = summary_num_results
        self._summary_prompt_name = summary_prompt_name
        self._node_postprocessors = node_postprocessors or []
        self._verbose = verbose

        self.callback_manager = callback_manager or CallbackManager([])
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = self.callback_manager

        self.conv_id = None

    @classmethod
    def from_args(
        cls,
        retriever: VectaraRetriever,
        streaming: bool = False,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        **kwargs: Any,
    ) -> "VectaraChatEngine":
        """Initialize a ContextChatEngine from default parameters."""
        node_postprocessors = node_postprocessors or []
        return cls(
            retriever,
            streaming,
            node_postprocessors=node_postprocessors,
            callback_manager=Settings.callback_manager,
            **kwargs,
        )

    def chat(self, message: str) -> AgentChatResponse:
        """Chat with the agent."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: message}
        ) as query_event:
            kwargs = (
                {
                    "response_language": self._summary_response_lang,
                    "max_used_search_results": self._summary_num_results,
                    "generation_preset_name": self._summary_prompt_name,
                }
                if self._summary_enabled
                else {}
            )
            nodes, summary, self.conv_id = self._retriever._vectara_query(
                QueryBundle(message),
                chat=True,
                conv_id=self.conv_id,
                verbose=self._verbose,
                **kwargs,
            )
            query_event.on_end(payload={EventPayload.RESPONSE: summary["text"]})
            return AgentChatResponse(
                response=summary["text"],
                source_nodes=nodes,
                metadata={"fcs": summary.get("fcs", None)},
            )

    async def achat(self, message: str) -> AgentChatResponse:
        """Chat with the agent asynchronously."""
        return await self.chat(message)

    def set_chat_id(self, source_nodes: List, metadata: Dict) -> None:
        """Callback function for setting the conv_id."""
        self.conv_id = metadata.get("chat_id", self.conv_id)

    def stream_chat(self, message: str) -> StreamingAgentChatResponse:
        query_bundle = QueryBundle(message)

        return self._retriever._vectara_stream(
            query_bundle,
            chat=True,
            conv_id=self.conv_id,
            callback_func=self.set_chat_id,
        )

    async def astream_chat(self, message: str) -> StreamingAgentChatResponse:
        return await self.stream_chat(message)

    def reset(self) -> None:
        self.conv_id = None

    def chat_history(self) -> List[str]:
        return ["Not implemented Yet."]
