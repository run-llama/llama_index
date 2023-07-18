"""Base agent type."""
from typing import List, Optional
from llama_index.chat_engine.types import BaseChatEngine, StreamingAgentChatResponse
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.llms.base import ChatMessage
from llama_index.response.schema import RESPONSE_TYPE, Response


class BaseAgent(BaseChatEngine, BaseQueryEngine):
    """Base Agent."""

    # ===== Query Engine Interface =====
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        agent_response = self.chat(
            query_bundle.query_str,
            chat_history=[],
        )
        return Response(response=str(agent_response))

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        agent_response = await self.achat(
            query_bundle.query_str,
            chat_history=[],
        )
        return Response(response=str(agent_response))

    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("stream_chat not implemented")

    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("astream_chat not implemented")
