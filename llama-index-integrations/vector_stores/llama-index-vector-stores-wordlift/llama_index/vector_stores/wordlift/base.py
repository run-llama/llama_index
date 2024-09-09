import asyncio
import json
from logging import getLogger
from typing import Any, List, Dict, Optional

import nest_asyncio
import wordlift_client
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores import (
    VectorStoreQuery,
    VectorStoreQueryResult,
    MetadataFilters,
)
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
)
from wordlift_client import NodeRequest, VectorSearchQueryRequest, Configuration
from wordlift_client.exceptions import ApiException
from wordlift_client.models import AccountInfo, NodeRequestMetadataValue

from .metadata_filters_to_filters import MetadataFiltersToFilters

log = getLogger(__name__)


def _make_configuration(
    host: str = "https://api.wordlift.io", key: str = None
) -> Configuration:
    """
    Create a Configuration instance to provide to an ApiClient.

    :param host: The api endpoint, by default https://api.wordlift.io
    :param key: The API key
    :return: A Configuration instance
    """
    configuration = Configuration(
        host=host,
    )

    configuration.api_key["ApiKey"] = key
    configuration.api_key_prefix["ApiKey"] = "Key"

    return configuration


def _generate_id(account: AccountInfo, node_id: str) -> str:
    return _trailing_slash(account.dataset_uri) + node_id


def _trailing_slash(value: str) -> str:
    if not value.endswith("/"):
        value += "/"
    return value


def _make_metadata_as_node_request_metadata_value(
    metadata: Dict[str, Any]
) -> Dict[str, NodeRequestMetadataValue]:
    values: Dict[str, NodeRequestMetadataValue] = {}
    for key, value in metadata.items():
        values[key] = NodeRequestMetadataValue(value)

    return values


class WordliftVectorStore(BasePydanticVectorStore):
    stores_text: bool = True

    _account: Optional[AccountInfo] = PrivateAttr(default=None)
    _configuration: Configuration = PrivateAttr()
    _fields: Optional[List[str]] = PrivateAttr()

    def __init__(
        self,
        key: Optional[str] = None,
        configuration: Optional[Configuration] = None,
        fields: Optional[List[str]] = None,
    ):
        super().__init__(use_async=True)
        nest_asyncio.apply()

        if configuration is None:
            self._configuration = _make_configuration(key=key)
        else:
            self._configuration = configuration

        if fields is None:
            self._fields = ["schema:url", "schema:name"]
        else:
            self._fields = fields

    @property
    def account(self) -> AccountInfo:
        if self._account is None:
            self._account = asyncio.get_event_loop().run_until_complete(
                self._get_account()
            )

        return self._account

    async def _get_account(self):
        """
        Get the account data for the provided key.

        :return:
        """
        async with wordlift_client.ApiClient(self._configuration) as api_client:
            api_instance = wordlift_client.AccountApi(api_client)

            try:
                return await api_instance.get_me()
            except ApiException as e:
                raise RuntimeError(
                    "Failed to get account info, check the provided key"
                ) from e

    @property
    def client(self) -> Any:
        return self.account

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        return asyncio.get_event_loop().run_until_complete(
            self.async_add(nodes, **add_kwargs)
        )

    async def async_add(
        self,
        nodes: List[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        # Empty nodes, return empty list
        if not nodes:
            return []

        log.debug(f"{len(nodes)} node(s) received\n")

        requests = []
        for node in nodes:
            node_dict = node.dict()
            # metadata: Dict[str, Any] = node_dict.get("metadata", {})
            metadata = _make_metadata_as_node_request_metadata_value(
                node_dict.get("metadata", {})
            )

            # Get or generate an ID
            entity_id = metadata.get("entity_id", None)
            if entity_id is None:
                entity_id = _generate_id(self.account, node.id_)

            entry = NodeRequest(
                entity_id=entity_id,
                node_id=node.node_id,
                embeddings=node.get_embedding(),
                text=node.get_content(metadata_mode=MetadataMode.NONE) or "",
                metadata=metadata,
            )
            requests.append(entry)

        async with wordlift_client.ApiClient(self._configuration) as api_client:
            api_instance = wordlift_client.VectorSearchNodesApi(api_client)

            try:
                await api_instance.update_nodes_collection(node_request=requests)
            except ApiException as e:
                raise RuntimeError("Error creating entities") from e

        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        return asyncio.get_event_loop().run_until_complete(
            self.adelete(ref_doc_id, **delete_kwargs)
        )

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        await self.adelete_nodes([ref_doc_id], **delete_kwargs)

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        return asyncio.get_event_loop().run_until_complete(
            self.adelete_nodes(node_ids, filters, **delete_kwargs)
        )

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        # Bail out if the list is not provided.
        if node_ids is None:
            return

        # Create the IDs.
        ids = []
        for node_id in node_ids:
            ids.append(_generate_id(self.account, node_id))

        async with wordlift_client.ApiClient(self._configuration) as api_client:
            api_instance = wordlift_client.EntitiesApi(api_client)

            try:
                await api_instance.delete_entities(id=ids)
            except ApiException as e:
                raise RuntimeError("Error deleting entities") from e

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        return asyncio.get_event_loop().run_until_complete(self.aquery(query, **kwargs))

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        filters = MetadataFiltersToFilters.metadata_filters_to_filters(
            query.filters if query.filters else []
        )
        if query.query_str:
            request = VectorSearchQueryRequest(
                query_string=query.query_str,
                similarity_top_k=query.similarity_top_k,
                fields=self._fields,
                filters=filters,
            )
        else:
            request = VectorSearchQueryRequest(
                query_embedding=query.query_embedding,
                similarity_top_k=query.similarity_top_k,
                fields=self._fields,
                filters=filters,
            )

        async with wordlift_client.ApiClient(self._configuration) as api_client:
            api_instance = wordlift_client.VectorSearchQueriesApi(api_client)

            try:
                page = await api_instance.create_query(
                    vector_search_query_request=request,
                )
            except ApiException as e:
                log.error(
                    f"Error querying for entities with the following request: {json.dumps(api_client.sanitize_for_serialization(request))}",
                    exc_info=True,
                )

        nodes: List[TextNode] = []
        similarities: List[float] = []
        ids: List[str] = []

        for item in page.items:
            metadata = item.metadata if item.metadata else {}
            fields = item.fields if item.fields else {}
            metadata = {**metadata, **fields}

            nodes.append(
                TextNode(
                    text=item.text if item.text else "",
                    id_=item.node_id if item.node_id else "",
                    embedding=(item.embeddings if "embeddings" in item else None),
                    metadata=metadata,
                )
            )
            similarities.append(item.score)
            ids.append(item.node_id)
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
