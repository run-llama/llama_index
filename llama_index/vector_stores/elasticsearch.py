"""Elasticsearch vector store."""
import uuid
from logging import getLogger
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union, cast

from llama_index.schema import MetadataMode, TextNode
from llama_index.vector_stores.types import (
    MetadataFilters,
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

logger = getLogger(__name__)

DISTANCE_STRATEGIES = Literal[
    "COSINE",
    "DOT_PRODUCT",
    "EUCLIDEAN_DISTANCE",
]


def _get_elasticsearch_client(
    *,
    es_url: Optional[str] = None,
    cloud_id: Optional[str] = None,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Any:
    """Get Elasticsearch client.

    Args:
        es_url: Elasticsearch URL.
        cloud_id: Elasticsearch cloud ID.
        api_key: Elasticsearch API key.
        username: Elasticsearch username.
        password: Elasticsearch password.

    Returns:
        Elasticsearch client.

    Raises:
        ConnectionError: If Elasticsearch client cannot connect to Elasticsearch.
    """

    try:
        import elasticsearch
    except ImportError:
        raise ImportError(
            "Could not import elasticsearch python package. "
            "Please install it with `pip install elasticsearch`."
        )

    if es_url and cloud_id:
        raise ValueError(
            "Both es_url and cloud_id are defined. Please provide only one."
        )

    if es_url and cloud_id:
        raise ValueError(
            "Both es_url and cloud_id are defined. Please provide only one."
        )

    connection_params: Dict[str, Any] = {}

    if es_url:
        connection_params["hosts"] = [es_url]
    elif cloud_id:
        connection_params["cloud_id"] = cloud_id
    else:
        raise ValueError("Please provide either elasticsearch_url or cloud_id.")

    if api_key:
        connection_params["api_key"] = api_key
    elif username and password:
        connection_params["basic_auth"] = (username, password)

    es_client = elasticsearch.Elasticsearch(**connection_params)
    try:
        es_client.info()
    except Exception as e:
        logger.error(f"Error connecting to Elasticsearch: {e}")
        raise e

    return es_client


def _to_elasticsearch_filter(standard_filters: MetadataFilters) -> Dict[str, Any]:
    """Convert standard filters to Elasticsearch filter.

    Args:
        standard_filters: Standard Llama-index filters.

    Returns:
        Elasticsearch filter.
    """
    if len(standard_filters.filters) == 1:
        filter = standard_filters.filters[0]
        return {
            "term": {
                f"metadata.{filter.key}": {
                    "value": filter.value,
                }
            }
        }
    else:
        operands = []
        for filter in standard_filters.filters:
            operands.append(
                {
                    "term": {
                        f"metadata.{filter.key}": {
                            "value": filter.value,
                        }
                    }
                }
            )
        return {"bool": {"must": operands}}


class ElasticsearchStore(VectorStore):
    """Elasticsearch vector store.

    Args:

        index_name: Name of the Elasticsearch index.
        es_client: Optional. Pre-existing Elasticsearch client.
        es_url: Optional. Elasticsearch URL.
        es_cloud_id: Optional. Elasticsearch cloud ID.
        es_api_key: Optional. Elasticsearch API key.
        es_user: Optional. Elasticsearch username.
        es_password: Optional. Elasticsearch password.
        text_field: Optional. Name of the Elasticsearch field that stores the text.
        vector_field: Optional. Name of the Elasticsearch field that stores the embedding.
        batch_size: Optional. Batch size for bulk indexing. Defaults to 200.
        distance_strategy: Optional. Distance strategy to use for similarity search. Defaults to "COSINE".

    Raises:
        ConnectionError: If Elasticsearch client cannot connect to Elasticsearch.
        ValueError: If neither es_client nor es_url nor es_cloud_id is provided.

    """

    stores_text: bool = True

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
    ) -> None:
        self.index_name = index_name
        self.text_field = text_field
        self.vector_field = vector_field
        self.batch_size = batch_size
        self.distance_strategy = distance_strategy

        if es_client is not None:
            self._client = es_client
        elif es_url is not None or es_cloud_id is not None:
            self._client = _get_elasticsearch_client(
                es_url=es_url,
                username=es_user,
                password=es_password,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
            )
        else:
            raise ValueError(
                """Either provide a pre-existing Elasticsearch connection, \
                or valid credentials for creating a new connection."""
            )

    @property
    def client(self) -> Any:
        """Get elasticsearch client."""
        return self._client

    def _create_index_if_not_exists(
        self, index_name: str, dims_length: Optional[int] = None
    ) -> None:
        """Create the Elasticsearch index if it doesn't already exist.

        Args:
            index_name: Name of the Elasticsearch index to create.
            dims_length: Length of the embedding vectors.
        """

        if self.client.indices.exists(index=index_name):
            logger.debug(f"Index {index_name} already exists. Skipping creation.")

        else:
            if dims_length is None:
                raise ValueError(
                    "Cannot create index without specifying dims_length "
                    "when the index doesn't already exist. We infer "
                    "dims_length from the first embedding. Check that "
                    "you have provided an embedding function."
                )

            if self.distance_strategy == "COSINE":
                similarityAlgo = "cosine"
            elif self.distance_strategy == "EUCLIDEAN_DISTANCE":
                similarityAlgo = "l2_norm"
            elif self.distance_strategy == "DOT_PRODUCT":
                similarityAlgo = "dot_product"
            else:
                raise ValueError(f"Similarity {self.distance_strategy} not supported.")

            index_settings = {
                "mappings": {
                    "properties": {
                        self.vector_field: {
                            "type": "dense_vector",
                            "dims": dims_length,
                            "index": True,
                            "similarity": similarityAlgo,
                        },
                    }
                }
            }

            logger.debug(
                f"Creating index {index_name} with mappings {index_settings['mappings']}"
            )
            self.client.indices.create(index=index_name, **index_settings)

    def add(
        self,
        embedding_results: List[NodeWithEmbedding],
        *,
        create_index_if_not_exists: bool = True,
    ) -> List[str]:
        """Add nodes to Elasticsearch index.

        Args:
            embedding_results: List of nodes with embeddings.
            create_index_if_not_exists: Optional. Whether to create the Elasticsearch index if it doesn't already exist. Defaults to True.

        Returns:
            List of node IDs that were added to the index.

        Raises:
            ImportError: If elasticsearch python package is not installed.
            BulkIndexError: If Elasticsearch bulk indexing fails.
        """
        try:
            from elasticsearch.helpers import BulkIndexError, bulk
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )

        if len(embedding_results) == 0:
            return []

        if create_index_if_not_exists:
            dims_length = len(embedding_results[0].embedding)
            self._create_index_if_not_exists(
                index_name=self.index_name, dims_length=dims_length
            )

        embeddings: List[List[float]] = []
        texts: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []
        for node in embedding_results:
            ids.append(node.id)
            embeddings.append(node.embedding)
            texts.append(node.node.get_content(metadata_mode=MetadataMode.NONE))
            metadatas.append(node_to_metadata_dict(node.node, remove_text=True))

        requests = []
        return_ids = []

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            _id = ids[i] if ids else str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                self.vector_field: embeddings[i],
                self.text_field: text,
                "metadata": metadata,
                "_id": _id,
            }
            requests.append(request)
            return_ids.append(_id)

        bulk(self.client, requests, chunk_size=self.batch_size, refresh=True)
        try:
            success, failed = bulk(self.client, requests, stats_only=True, refresh=True)
            logger.debug(f"Added {success} and failed to add {failed} texts to index")

            logger.debug(f"added texts {ids} to index")
            return return_ids
        except BulkIndexError as e:
            logger.error(f"Error adding texts: {e}")
            firstError = e.errors[0].get("index", {}).get("error", {})
            logger.error(f"First error reason: {firstError.get('reason')}")
            raise e

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete node from Elasticsearch index.

        Args:
            ref_doc_id: ID of the node to delete.
            delete_kwargs: Optional. Additional arguments to pass to Elasticsearch delete_by_query.

        Raises:
            Exception: If Elasticsearch delete_by_query fails.
        """

        try:
            self.client.delete_by_query(
                index=self.index_name,
                query={"match": {"_id": ref_doc_id}},
                refresh=True,
                **delete_kwargs,
            )
            logger.debug(f"Deleted text {ref_doc_id} from index")
        except Exception as e:
            logger.error(f"Error deleting text: {ref_doc_id}")
            raise e

    def query(
        self,
        query: VectorStoreQuery,
        custom_query: Optional[
            Callable[[Dict, Union[VectorStoreQuery, None]], Dict]
        ] = None,
        es_filter: Optional[Dict] = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            custom_query: Optional. custom query function that takes in the es query body
                            and returns a modified query body. This can be used to add
                            additional query parameters to the Elasticsearch query.
            es_filter: Optional. Elasticsearch filter to apply to the query. If filter is provided
                    in the query, this filter will be ignored.

        Returns:
            VectorStoreQueryResult: Result of the query.

        Raises:
            Exception: If Elasticsearch query fails.

        """
        query_embedding = cast(List[float], query.query_embedding)

        es_query = {}

        if query.filters is not None and len(query.filters.filters) > 0:
            filter = [_to_elasticsearch_filter(query.filters)]
        else:
            filter = es_filter or []

        if query.mode in (
            VectorStoreQueryMode.DEFAULT,
            VectorStoreQueryMode.HYBRID,
        ):
            es_query["knn"] = {
                "filter": filter,
                "field": self.vector_field,
                "query_vector": query_embedding,
                "k": query.similarity_top_k,
                "num_candidates": query.similarity_top_k * 10,
            }

        if query.mode in (
            VectorStoreQueryMode.TEXT_SEARCH,
            VectorStoreQueryMode.HYBRID,
        ):
            es_query["query"] = {
                "bool": {
                    "must": {"match": {self.text_field: {"query": query.query_str}}},
                    "filter": filter,
                }
            }

        if query.mode == VectorStoreQueryMode.HYBRID:
            es_query["rank"] = {"rrf": {}}

        if custom_query is not None:
            es_query = custom_query(es_query, query)
            logger.debug(f"Calling custom_query, Query body now: {es_query}")

        response = self.client.search(
            index=self.index_name,
            **es_query,
            size=query.similarity_top_k,
            _source={"excludes": [self.vector_field]},
        )

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            metadata = source.get("metadata", None)
            text = source.get(self.text_field, None)
            node_id = hit["_id"]

            try:
                node = metadata_dict_to_node(metadata)
                node.text = text
            except Exception:
                # Legacy support for old metadata format
                logger.warning(
                    f"Could not parse metadata from hit {hit['_source']['metadata']}"
                )
                node_info = source.get("node_info")
                relationships = source.get("relationships")
                start_char_idx = None
                end_char_idx = None
                if isinstance(node_info, dict):
                    start_char_idx = node_info.get("start", None)
                    end_char_idx = node_info.get("end", None)

                node = TextNode(
                    text=text,
                    metadata=metadata,
                    id_=node_id,
                    start_char_idx=start_char_idx,
                    end_char_idx=end_char_idx,
                    relationships=relationships,
                )
            top_k_nodes.append(node)
            top_k_ids.append(node_id)
            top_k_scores.append(hit["_score"])
        return VectorStoreQueryResult(
            nodes=top_k_nodes, ids=top_k_ids, similarities=top_k_scores
        )
