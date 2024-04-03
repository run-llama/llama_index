import asyncio
import logging
import traceback
from typing import Any, List, Dict, Union

from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.vector_stores.types import (
    VectorStore,
)

from manager_client import NodeRequest, VectorSearchQueryRequest
from llama_index.vector_stores.wordlift.utils import VectorSearchService

log = logging.getLogger("global")


class KeyProvider:
    key: str

    def __init__(self, key: str):
        self.key = key

    def for_add(self, nodes: List[BaseNode]) -> str:
        return self.key

    def for_delete(self, ref_doc_id: str) -> str:
        return self.key

    def for_query(self, query: VectorStoreQuery) -> str:
        return self.key


class WordliftVectorStore(VectorStore):
    stores_text = True

    key: str
    vector_search_service: VectorSearchService

    @staticmethod
    def create(key: str):
        return WordliftVectorStore(KeyProvider(key), VectorSearchService())

    def __init__(
        self,
        key_provider: KeyProvider,
        vector_search_service: VectorSearchService,
    ):
        super(WordliftVectorStore, self).__init__(use_async=True)

        self.vector_search_service = vector_search_service
        self.key_provider = key_provider

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        log.debug("Add node(s)\n")
        loop = asyncio.get_event_loop()
        task = loop.create_task(self.async_add(nodes, **add_kwargs))
        return loop.run_until_complete(task)

    async def async_add(
        self,
        nodes: List[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        # Empty nodes, return empty list
        if not nodes:
            return []

        log.debug(f"{len(nodes)} node(s) received\n")

        # Get the key to use for the operation.
        key = self.key_provider.for_add(nodes)

        requests = []
        for node in nodes:
            node_dict = node.dict()
            metadata: Dict[str, Any] = node_dict.get("metadata", {})
            entity_id = metadata.get("entity_id", None)

            entry = NodeRequest(
                entity_id=entity_id,
                node_id=node.node_id,
                embeddings=node.get_embedding(),
                text=node.get_content(metadata_mode=MetadataMode.NONE) or "",
                metadata={},
            )
            requests.append(entry)

        log.debug(f"Inserting data, using key {key}: {requests}")

        try:
            await self.vector_search_service.update_nodes_collection(
                node_request=requests, key=key
            )
        except Exception:
            print(traceback.format_exc())
            return []

        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError

    async def query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        log.debug("Running in NON async mode")
        loop = asyncio.get_event_loop()
        task = loop.create_task(self.aquery(query, **kwargs))
        return loop.run_until_complete(task)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        dataset_id: Union[str, None] = kwargs.get("dataset_id", None)
        if dataset_id is None:
            raise ValueError("Missing dataset_id value.")

        request = VectorSearchQueryRequest(
            query_embedding=query.query_embedding,
            similarity_top_k=query.similarity_top_k,
        )

        # Get the key to use for the operation.
        key = self.key_provider.for_query(query)

        try:
            page = await self.vector_search_service.query_nodes_collection(
                vector_search_query_request=request, key=key
            )
        except Exception:
            print(traceback.format_exc())
            raise Exception("Failed to fetch query results")

        nodes: List[TextNode] = []
        similarities: List[float] = []
        ids: List[str] = []

        for item in page.items:
            nodes.append(
                TextNode(
                    text=item.text,
                    id_=item.node_id,
                    embedding=item.embeddings,
                    metadata=item.metadata,
                )
            )
            similarities.append(item.score)
            ids.append(item.node_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
