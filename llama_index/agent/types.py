"""Base agent type."""
from typing import List, Optional

from llama_index.callbacks import trace_method
from llama_index.chat_engine.types import BaseChatEngine, StreamingAgentChatResponse
from llama_index.core import BaseQueryEngine
from llama_index.llms.base import ChatMessage
from llama_index.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.schema import QueryBundle


class BaseAgent(BaseChatEngine, BaseQueryEngine):
    """Base Agent."""

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        # TODO: the ReAct agent does not explicitly specify prompts, would need a
        # refactor to expose those prompts
        return {}

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    # ===== Query Engine Interface =====
    @trace_method("query")
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        agent_response = self.chat(
            query_bundle.query_str,
            chat_history=[],
        )
        return Response(
            response=str(agent_response), source_nodes=agent_response.source_nodes
        )

    @trace_method("query")
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        agent_response = await self.achat(
            query_bundle.query_str,
            chat_history=[],
        )
        return Response(
            response=str(agent_response), source_nodes=agent_response.source_nodes
        )

    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("stream_chat not implemented")

    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("astream_chat not implemented")
