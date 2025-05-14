"""Elasticsearch vector store."""

import asyncio
from logging import getLogger
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import nest_asyncio
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
)
from elasticsearch.helpers.vectorstore import AsyncVectorStore
from elasticsearch.helpers.vectorstore import (
    AsyncBM25Strategy,
    AsyncSparseVectorStrategy,
    AsyncDenseVectorStrategy,
    AsyncRetrievalStrategy,
    DistanceMetric,
)

from llama_index.vector_stores.elasticsearch.utils import (
    get_elasticsearch_client,
    get_user_agent,
    convert_es_hit_to_node,
)

logger = getLogger(__name__)

DISTANCE_STRATEGIES = Literal[
    "COSINE",
    "DOT_PRODUCT",
    "EUCLIDEAN_DISTANCE",
]


def _to_elasticsearch_filter(
    standard_filters: MetadataFilters, metadata_keyword_suffix: str = ".keyword"
) -> Dict[str, Any]:
    """
    Convert standard filters to Elasticsearch filter.

    Args:
        standard_filters: Standard Llama-index filters.

    Returns:
        Elasticsearch filter.

    """
    if len(standard_filters.legacy_filters()) == 1:
        filter = standard_filters.legacy_filters()[0]
        return {
            "term": {
                f"metadata.{filter.key}{metadata_keyword_suffix}": {
                    "value": filter.value,
                }
            }
        }
    else:
        operands = []
        for filter in standard_filters.legacy_filters():
            operands.append(
                {
                    "term": {
                        f"metadata.{filter.key}{metadata_keyword_suffix}": {
                            "value": filter.value,
                        }
                    }
                }
            )
        return {"bool": {"must": operands}}


def _to_llama_similarities(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 if max_score > 0 else 0.0 for _ in scores]
    return [(x - min_score) / (max_score - min_score) for x in scores]


def _mode_must_match_retrieval_strategy(
    mode: VectorStoreQueryMode, retrieval_strategy: AsyncRetrievalStrategy
) -> None:
    """
    Different retrieval strategies require different ways of indexing that must be known at the
    time of adding data. The query mode is known at query time. This function checks if the
    retrieval strategy (and way of indexing) is compatible with the query mode and raises and
    exception in the case of a mismatch.
    """
    if mode == VectorStoreQueryMode.DEFAULT:
        # it's fine to not specify an explicit other mode
        return

    mode_retrieval_dict = {
        VectorStoreQueryMode.SPARSE: AsyncSparseVectorStrategy,
        VectorStoreQueryMode.TEXT_SEARCH: AsyncBM25Strategy,
        VectorStoreQueryMode.HYBRID: AsyncDenseVectorStrategy,
    }

    required_strategy = mode_retrieval_dict.get(mode)
    if not required_strategy:
        raise NotImplementedError(f"query mode {mode} currently not supported")

    if not isinstance(retrieval_strategy, required_strategy):
        raise ValueError(
            f"query mode {mode} incompatible with retrieval strategy {type(retrieval_strategy)}, "
            f"expected {required_strategy}"
        )

    if mode == VectorStoreQueryMode.HYBRID and not retrieval_strategy.hybrid:
        raise ValueError(f"to enable hybrid mode, it must be set in retrieval strategy")


class ElasticsearchStore(BasePydanticVectorStore):
    """
    Elasticsearch vector store.

    Args:
        index_name: Name of the Elasticsearch index.
        es_client: Optional. Pre-existing AsyncElasticsearch client.
        es_url: Optional. Elasticsearch URL.
        es_cloud_id: Optional. Elasticsearch cloud ID.
        es_api_key: Optional. Elasticsearch API key.
        es_user: Optional. Elasticsearch username.
        es_password: Optional. Elasticsearch password.
        text_field: Optional. Name of the Elasticsearch field that stores the text.
        vector_field: Optional. Name of the Elasticsearch field that stores the
                    embedding.
        batch_size: Optional. Batch size for bulk indexing. Defaults to 200.
        distance_strategy: Optional. Distance strategy to use for similarity search.
                        Defaults to "COSINE".
        retrieval_strategy: Retrieval strategy to use. AsyncBM25Strategy /
            AsyncSparseVectorStrategy / AsyncDenseVectorStrategy / AsyncRetrievalStrategy.
            Defaults to AsyncDenseVectorStrategy.

    Raises:
        ConnectionError: If AsyncElasticsearch client cannot connect to Elasticsearch.
        ValueError: If neither es_client nor es_url nor es_cloud_id is provided.

    Examples:
        `pip install llama-index-vector-stores-elasticsearch`

        ```python
        from llama_index.vector_stores import ElasticsearchStore

        # Additional setup for ElasticsearchStore class
        index_name = "my_index"
        es_url = "http://localhost:9200"
        es_cloud_id = "<cloud-id>"  # Found within the deployment page
        es_user = "elastic"
        es_password = "<password>"  # Provided when creating deployment or can be reset
        es_api_key = "<api-key>"  # Create an API key within Kibana (Security -> API Keys)

        # Connecting to ElasticsearchStore locally
        es_local = ElasticsearchStore(
            index_name=index_name,
            es_url=es_url,
        )

        # Connecting to Elastic Cloud with username and password
        es_cloud_user_pass = ElasticsearchStore(
            index_name=index_name,
            es_cloud_id=es_cloud_id,
            es_user=es_user,
            es_password=es_password,
        )

        # Connecting to Elastic Cloud with API Key
        es_cloud_api_key = ElasticsearchStore(
            index_name=index_name,
            es_cloud_id=es_cloud_id,
            es_api_key=es_api_key,
        )
        ```

    """

    class Config:
        # allow pydantic to tolarate its inability to validate AsyncRetrievalStrategy
        arbitrary_types_allowed = True

    stores_text: bool = True
    index_name: str
    es_client: Optional[Any]
    es_url: Optional[str]
    es_cloud_id: Optional[str]
    es_api_key: Optional[str]
    es_user: Optional[str]
    es_password: Optional[str]
    text_field: str = "content"
    vector_field: str = "embedding"
    batch_size: int = 200
    distance_strategy: Optional[DISTANCE_STRATEGIES] = "COSINE"
    retrieval_strategy: AsyncRetrievalStrategy

    _store = PrivateAttr()

    def __init__(
        self,
        index_name: str,
        es_client: Optional[Any] = None,
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_user: Optional[str] = None,
        es_password: Optional[str] = None,
        text_field: str = "content",
        vector_field: str = "embedding",
        batch_size: int = 200,
        distance_strategy: Optional[DISTANCE_STRATEGIES] = "COSINE",
        retrieval_strategy: Optional[AsyncRetrievalStrategy] = None,
        metadata_mappings: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        nest_asyncio.apply()

        if not es_client:
            es_client = get_elasticsearch_client(
                url=es_url,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
                username=es_user,
                password=es_password,
            )

        if retrieval_strategy is None:
            retrieval_strategy = AsyncDenseVectorStrategy(
                distance=DistanceMetric[distance_strategy]
            )

        base_metadata_mappings = {
            "document_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "ref_doc_id": {"type": "keyword"},
        }

        metadata_mappings = metadata_mappings or {}
        metadata_mappings.update(base_metadata_mappings)

        super().__init__(
            index_name=index_name,
            es_client=es_client,
            es_url=es_url,
            es_cloud_id=es_cloud_id,
            es_api_key=es_api_key,
            es_user=es_user,
            es_password=es_password,
            text_field=text_field,
            vector_field=vector_field,
            batch_size=batch_size,
            distance_strategy=distance_strategy,
            retrieval_strategy=retrieval_strategy,
        )

        self._store = AsyncVectorStore(
            user_agent=get_user_agent(),
            client=es_client,
            index=index_name,
            retrieval_strategy=retrieval_strategy,
            text_field=text_field,
            vector_field=vector_field,
            metadata_mappings=metadata_mappings,
        )

        # Disable query embeddings when using Sparse vectors or BM25.
        # ELSER generates its own embeddings server-side
        if not isinstance(retrieval_strategy, AsyncDenseVectorStrategy):
            self.is_embedding_query = False

    @property
    def client(self) -> Any:
        """Get async elasticsearch client."""
        return self._store.client

    def close(self) -> None:
        return asyncio.get_event_loop().run_until_complete(self._store.close())

    def add(
        self,
        nodes: List[BaseNode],
        *,
        create_index_if_not_exists: bool = True,
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to Elasticsearch index.

        Args:
            nodes: List of nodes with embeddings.
            create_index_if_not_exists: Optional. Whether to create
                                        the Elasticsearch index if it
                                        doesn't already exist.
                                        Defaults to True.

        Returns:
            List of node IDs that were added to the index.

        Raises:
            ImportError: If elasticsearch['async'] python package is not installed.
            BulkIndexError: If AsyncElasticsearch async_bulk indexing fails.

        """
        return asyncio.get_event_loop().run_until_complete(
            self.async_add(
                nodes,
                create_index_if_not_exists=create_index_if_not_exists,
                **add_kwargs,
            )
        )

    async def async_add(
        self,
        nodes: List[BaseNode],
        *,
        create_index_if_not_exists: bool = True,
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Asynchronous method to add nodes to Elasticsearch index.

        Args:
            nodes: List of nodes with embeddings.
            create_index_if_not_exists: Optional. Whether to create
                                        the AsyncElasticsearch index if it
                                        doesn't already exist.
                                        Defaults to True.

        Returns:
            List of node IDs that were added to the index.

        Raises:
            ImportError: If elasticsearch python package is not installed.
            BulkIndexError: If AsyncElasticsearch async_bulk indexing fails.

        """
        if len(nodes) == 0:
            return []

        embeddings: Optional[List[List[float]]] = None
        texts: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []
        for node in nodes:
            ids.append(node.node_id)
            texts.append(node.get_content(metadata_mode=MetadataMode.NONE))
            metadatas.append(node_to_metadata_dict(node, remove_text=True))

        # Generate embeddings when using dense vectors. They are not needed
        # for other strategies.
        if isinstance(self.retrieval_strategy, AsyncDenseVectorStrategy):
            embeddings = []
            for node in nodes:
                embeddings.append(node.get_embedding())

            if not self._store.num_dimensions:
                self._store.num_dimensions = len(embeddings[0])

        return await self._store.add_texts(
            texts=texts,
            metadatas=metadatas,
            vectors=embeddings,
            ids=ids,
            create_index_if_not_exists=create_index_if_not_exists,
            bulk_kwargs=add_kwargs,
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete node from Elasticsearch index.

        Args:
            ref_doc_id: ID of the node to delete.
            delete_kwargs: Optional. Additional arguments to
                        pass to Elasticsearch delete_by_query.

        Raises:
            Exception: If Elasticsearch delete_by_query fails.

        """
        return asyncio.get_event_loop().run_until_complete(
            self.adelete(ref_doc_id, **delete_kwargs)
        )

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Async delete node from Elasticsearch index.

        Args:
            ref_doc_id: ID of the node to delete.
            delete_kwargs: Optional. Additional arguments to
                        pass to AsyncElasticsearch delete_by_query.

        Raises:
            Exception: If AsyncElasticsearch delete_by_query fails.

        """
        await self._store.delete(
            query={"term": {"metadata.ref_doc_id": ref_doc_id}}, **delete_kwargs
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Delete nodes from vector store using node IDs and filters.

        Args:
            node_ids: Optional list of node IDs to delete.
            filters: Optional metadata filters to select nodes to delete.
            delete_kwargs: Optional additional arguments to pass to delete operation.

        """
        return asyncio.get_event_loop().run_until_complete(
            self.adelete_nodes(node_ids, filters, **delete_kwargs)
        )

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Asynchronously delete nodes from vector store using node IDs and filters.

        Args:
            node_ids (Optional[List[str]], optional): List of node IDs. Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.
            delete_kwargs (Any, optional): Optional additional arguments to pass to delete operation.

        """
        if not node_ids and not filters:
            return

        if node_ids and not filters:
            await self._store.delete(ids=node_ids, **delete_kwargs)
            return

        query = {"bool": {"must": []}}

        if node_ids:
            query["bool"]["must"].append({"terms": {"_id": node_ids}})

        if filters:
            es_filter = _to_elasticsearch_filter(filters)
            if "bool" in es_filter and "must" in es_filter["bool"]:
                query["bool"]["must"].extend(es_filter["bool"]["must"])
            else:
                query["bool"]["must"].append(es_filter)

        await self._store.delete(query=query, **delete_kwargs)

    def query(
        self,
        query: VectorStoreQuery,
        custom_query: Optional[
            Callable[[Dict, Union[VectorStoreQuery, None]], Dict]
        ] = None,
        es_filter: Optional[List[Dict]] = None,
        metadata_keyword_suffix: str = ".keyword",
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            custom_query: Optional. custom query function that takes in the es query
                        body and returns a modified query body.
                        This can be used to add additional query
                        parameters to the Elasticsearch query.
            es_filter: Optional. Elasticsearch filter to apply to the
                        query. If filter is provided in the query,
                        this filter will be ignored.
            metadata_keyword_suffix (str): The suffix to append to the metadata field of the keyword type.

        Returns:
            VectorStoreQueryResult: Result of the query.

        Raises:
            Exception: If Elasticsearch query fails.

        """
        return asyncio.get_event_loop().run_until_complete(
            self.aquery(query, custom_query, es_filter, **kwargs)
        )

    async def aquery(
        self,
        query: VectorStoreQuery,
        custom_query: Optional[
            Callable[[Dict, Union[VectorStoreQuery, None]], Dict]
        ] = None,
        es_filter: Optional[List[Dict]] = None,
        metadata_keyword_suffix: str = ".keyword",
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Asynchronous query index for top k most similar nodes.

        Args:
            query_embedding (VectorStoreQuery): query embedding
            custom_query: Optional. custom query function that takes in the es query
                        body and returns a modified query body.
                        This can be used to add additional query
                        parameters to the AsyncElasticsearch query.
            es_filter: Optional. AsyncElasticsearch filter to apply to the
                        query. If filter is provided in the query,
                        this filter will be ignored.
            metadata_keyword_suffix (str): The suffix to append to the metadata field of the keyword type.

        Returns:
            VectorStoreQueryResult: Result of the query.

        Raises:
            Exception: If AsyncElasticsearch query fails.

        """
        _mode_must_match_retrieval_strategy(query.mode, self.retrieval_strategy)

        if query.filters is not None and len(query.filters.legacy_filters()) > 0:
            filter = [_to_elasticsearch_filter(query.filters, metadata_keyword_suffix)]
        else:
            filter = es_filter or []

        hits = await self._store.search(
            query=query.query_str,
            query_vector=query.query_embedding,
            k=query.similarity_top_k,
            num_candidates=query.similarity_top_k * 10,
            filter=filter,
            custom_query=custom_query,
        )

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []

        for hit in hits:
            node = convert_es_hit_to_node(hit, self.text_field)
            top_k_nodes.append(node)
            top_k_ids.append(hit["_id"])
            top_k_scores.append(hit["_score"])

        return VectorStoreQueryResult(
            nodes=top_k_nodes,
            ids=top_k_ids,
            similarities=_to_llama_similarities(top_k_scores),
        )

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """
        Get nodes from Elasticsearch index.

        Args:
            node_ids (Optional[List[str]]): List of node IDs to retrieve.
            filters (Optional[MetadataFilters]): Metadata filters to apply.

        Returns:
            List[BaseNode]: List of nodes retrieved from the index.

        """
        return asyncio.get_event_loop().run_until_complete(
            self.aget_nodes(node_ids, filters)
        )

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """
        Asynchronously get nodes from Elasticsearch index.

        Args:
            node_ids (Optional[List[str]]): List of node IDs to retrieve.
            filters (Optional[MetadataFilters]): Metadata filters to apply.

        Returns:
            List[BaseNode]: List of nodes retrieved from the index.

        Raises:
            ValueError: If neither node_ids nor filters is provided.

        """
        if not node_ids and not filters:
            raise ValueError("Either node_ids or filters must be provided.")

        query = {"bool": {"must": []}}

        if node_ids is not None:
            query["bool"]["must"].append({"terms": {"_id": node_ids}})

        if filters:
            es_filter = _to_elasticsearch_filter(filters)
            if "bool" in es_filter and "must" in es_filter["bool"]:
                query["bool"]["must"].extend(es_filter["bool"]["must"])
            else:
                query["bool"]["must"].append(es_filter)

        response = await self._store.client.search(
            index=self.index_name,
            body={"query": query, "size": 10000},
        )

        hits = response.get("hits", {}).get("hits", [])
        nodes = []

        for hit in hits:
            nodes.append(convert_es_hit_to_node(hit, self.text_field))

        return nodes

    def clear(self) -> None:
        """
        Clear all nodes from Elasticsearch index.
        This method deletes and recreates the index.
        """
        return asyncio.get_event_loop().run_until_complete(self.aclear())

    async def aclear(self) -> None:
        """
        Asynchronously clear all nodes from Elasticsearch index.
        This method deletes and recreates the index.
        """
        if await self._store.client.indices.exists(index=self.index_name):
            await self._store.client.indices.delete(index=self.index_name)
