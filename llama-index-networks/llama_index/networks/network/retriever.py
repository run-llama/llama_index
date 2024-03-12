import asyncio
from typing import List, Optional
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.async_utils import run_async_tasks
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.networks.contributor.retriever import ContributorRetrieverClient
from functools import reduce


class NetworkRetriever(BaseRetriever):
    """The network Retriever."""

    def __init__(
        self,
        contributors: List[ContributorRetrieverClient],
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._contributors = contributors
        super().__init__(callback_manager=callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        async_tasks = [
            contributor.aretrieve(query_bundle) for contributor in self._contributors
        ]
        contributor_results: List[List[NodeWithScore]] = run_async_tasks(async_tasks)
        return reduce(lambda x, y: x + y, contributor_results)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        async_tasks = [
            contributor.aretrieve(query_bundle) for contributor in self._contributors
        ]
        contributor_results: List[List[NodeWithScore]] = await asyncio.gather(
            *async_tasks
        )
        return reduce(lambda x, y: x + y, contributor_results)
