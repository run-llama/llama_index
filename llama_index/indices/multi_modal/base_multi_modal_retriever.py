import asyncio
from typing import List

from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.schema import QueryBundle, QueryType

# from llama_index.constants import DEFAULT_SIMILARITY_TOP_K, DEFAULT_IMAGE_SIMILARITY_TOP_K
from llama_index.retrievers.image_retriever import BaseImageRetriever
from llama_index.schema import NodeWithScore


class MultiModalRetriever(BaseRetriever, BaseImageRetriever):
    """Multi Modal base retriever."""

    def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return self._retrieve(str_or_query_bundle)

    # Retrieve text and image nodes and mixer them
    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        res = self._text_retrieve(query_bundle)
        res.extend(self._image_retrieve(query_bundle))
        return res

    def _text_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        return super()._retrieve(query_bundle)

    def text_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        return self._text_retrieve(query_bundle)

    def _image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        return super()._image_retrieve(query_bundle)

    def image_retrive(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._image_retrieve(query_bundle)

    # Async Methods

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Run the two retrievals in async, and return their results as a concatenated list
        results: List[NodeWithScore] = []
        tasks = [
            self._atext_retrieve(query_bundle),
            self._aimage_retrieve(query_bundle),
        ]

        task_results = await asyncio.gather(*tasks)

        for task_result in task_results:
            results.extend(task_result)
        return results

    async def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return await self._aretrieve(str_or_query_bundle)

    async def _atext_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        return await super()._aretrieve(query_bundle)

    async def atext_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return await self._atext_retrieve(str_or_query_bundle)

    async def _aimage_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        return await super()._aimage_retrieve(query_bundle)

    async def aimage_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return await self._aimage_retrieve(str_or_query_bundle)
