import asyncio
from typing import List, Optional
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
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
        reranker: Optional[BaseNodePostprocessor] = None,
        rerank: bool = False,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._contributors = contributors
        self.rerank = rerank
        self.reranker = reranker
        if self.rerank:
            if self.reranker is None:
                raise ValueError(
                    "If `rerank = True`, then `reranker` must not be `None`."
                )
        super().__init__(callback_manager=callback_manager)

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        return self.reranker.postprocess_nodes(nodes=nodes, query_bundle=query_bundle)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        async_tasks = [
            contributor.aretrieve(query_bundle) for contributor in self._contributors
        ]
        contributor_results: List[List[NodeWithScore]] = run_async_tasks(async_tasks)
        flattened_results = reduce(lambda x, y: x + y, contributor_results)
        if self.rerank:
            return self._postprocess_nodes(
                nodes=flattened_results, query_bundle=query_bundle
            )
        return flattened_results

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        async_tasks = [
            contributor.aretrieve(query_bundle) for contributor in self._contributors
        ]
        contributor_results: List[List[NodeWithScore]] = await asyncio.gather(
            *async_tasks
        )
        flattened_results = reduce(lambda x, y: x + y, contributor_results)
        if self.rerank:
            return self._postprocess_nodes(
                nodes=flattened_results, query_bundle=query_bundle
            )
        return flattened_results
