from typing import List, Optional
import asyncio

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.async_utils import run_async_tasks
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.networks.contributor.query_engine import ContributorClient


class NetworkRetriever(BaseRetriever):
    """The network Retriever."""

    def __init__(
        self,
        contributors: List[ContributorClient],
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._contributors = contributors
        super().__init__(callback_manager=callback_manager)

    def _retrieve(self, query: str) -> List[NodeWithScore]:
        results = []
        async_tasks = [contributor.aquery(query) for contributor in self._contributors]
        results = run_async_tasks(async_tasks)

        return [
            NodeWithScore(node=TextNode(text=el.response), score=el.metadata["score"])
            for el in results
        ]

    async def _aretrieve(self, query: str) -> List[NodeWithScore]:
        async_tasks = [contributor.aquery(query) for contributor in self._contributors]
        results = await asyncio.gather(*async_tasks)

        return [
            NodeWithScore(node=TextNode(text=el.response), score=el.metadata["score"])
            for el in results
        ]
