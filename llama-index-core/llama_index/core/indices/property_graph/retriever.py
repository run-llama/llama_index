import asyncio
from tqdm import tqdm
from typing import Any, List

from llama_index.core.async_utils import run_jobs
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.property_graph.sub_retrievers.base import (
    BaseLPGRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle


class LPGRetriever(BaseRetriever):
    def __init__(
        self,
        sub_retrievers: List[BaseLPGRetriever],
        num_workers: int = 4,
        use_async: bool = True,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        self.sub_retrievers = sub_retrievers
        self.use_async = use_async
        self.num_workers = num_workers
        self.show_progress = show_progress

    def _deduplicate(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        seen = set()
        deduped = []
        for node in nodes:
            if node.text not in seen:
                deduped.append(node)
                seen.add(node.text)

        return deduped

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        results = []
        if self.use_async:
            return asyncio.run(self._aretrieve(query_bundle))

        for sub_retriever in tqdm(self.sub_retrievers, disable=not self.show_progress):
            results.extend(sub_retriever.retrieve(query_bundle))

        return self._deduplicate(results)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        tasks = []
        for sub_retriever in self.sub_retrievers:
            tasks.append(sub_retriever.aretrieve(query_bundle))

        async_results = await run_jobs(
            tasks, workers=self.num_workers, show_progress=self.show_progress
        )

        # flatten the results
        return self._deduplicate([node for nodes in async_results for node in nodes])
