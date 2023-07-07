"""Base agent type."""
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.response.schema import RESPONSE_TYPE


class BaseAgent(BaseChatEngine, BaseQueryEngine):
    """Base Agent."""

    # ===== Query Engine Interface =====
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self.chat(
            query_bundle.query_str,
            chat_history=[],
        )

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return await self.achat(
            query_bundle.query_str,
            chat_history=[],
        )
