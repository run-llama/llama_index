from typing import Any, List, Optional

from llama_index.core.async_utils import run_async_tasks
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.indices.managed.postgresml.retriever import PostgresMLRetriever


class PostgresMLQueryEngine(BaseQueryEngine):
    """Retriever query engine for PostgresML.

    Args:
        retriever (PostgresMLRetriever): A retriever object.
    """

    def __init__(
        self,
        retriever: PostgresMLRetriever,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        super().__init__(callback_manager=callback_manager)

    @classmethod
    def from_args(
        cls,
        retriever: PostgresMLRetriever,
        **kwargs: Any,
    ) -> "PostgresMLQueryEngine":
        """Initialize a PostgresMLQueryEngine object.".

        Args:
            retriever (PostgresMLRetriever): A PostgresML retriever object.
        """
        return cls(
            retriever=retriever,
        )

    def with_retriever(self, retriever: PostgresMLRetriever) -> "PostgresMLQueryEngine":
        return PostgresMLQueryEngine(
            retriever=retriever,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return run_async_tasks([self._do_query(query_bundle)])[0]

    async def _do_query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        # TODO: Unified RAG here please
        return Response()

    # TODO: Look into the prompting stuff
    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}
