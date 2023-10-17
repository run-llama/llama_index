"""Elasticsearch/Opensearch vector store."""
import json
import uuid
from typing import Any, Dict, Iterable, List, Optional, Union, cast

from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.vector_stores.types import (
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

IMPORT_OPENSEARCH_PY_ERROR = (
    "Could not import OpenSearch. Please install it with `pip install opensearch-py`."
)
MATCH_ALL_QUERY = {"match_all": {}}  # type: Dict


def _import_opensearch() -> Any:
    """Import OpenSearch if available, otherwise raise error."""
    try:
        from opensearchpy import OpenSearch
    except ImportError:
        raise ValueError(IMPORT_OPENSEARCH_PY_ERROR)
    return OpenSearch


def _import_bulk() -> Any:
    """Import bulk if available, otherwise raise error."""
    try:
        from opensearchpy.helpers import bulk
    except ImportError:
        raise ValueError(IMPORT_OPENSEARCH_PY_ERROR)
    return bulk


def _import_not_found_error() -> Any:
    """Import not found error if available, otherwise raise error."""
    try:
        from opensearchpy.exceptions import NotFoundError
    except ImportError:
        raise ValueError(IMPORT_OPENSEARCH_PY_ERROR)
    return NotFoundError


def _get_opensearch_client(opensearch_url: str, **kwargs: Any) -> Any:
    """Get OpenSearch client from the opensearch_url, otherwise raise error."""
    try:
        opensearch = _import_opensearch()
        client = opensearch(opensearch_url, **kwargs)
    except ValueError as e:
        raise ValueError(
            f"OpenSearch client string provided is not in proper format. "
            f"Got error: {e} "
        )
    return client


def _bulk_ingest_embeddings(
    client: Any,
    index_name: str,
    embeddings: List[List[float]],
    texts: Iterable[str],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    vector_field: str = "embedding",
    text_field: str = "content",
    mapping: Optional[Dict] = None,
    max_chunk_bytes: Optional[int] = 1 * 1024 * 1024,
) -> List[str]:
    """Bulk Ingest Embeddings into given index."""
    if not mapping:
        mapping = {}

    bulk = _import_bulk()
    not_found_error = _import_not_found_error()
    requests = []
    return_ids = []
    mapping = mapping

    try:
        client.indices.get(index=index_name)
    except not_found_error:
        client.indices.create(index=index_name, body=mapping)

    for i, text in enumerate(texts):
        metadata = metadatas[i] if metadatas else {}
        _id = ids[i] if ids else str(uuid.uuid4())
        request = {
            "_op_type": "index",
            "_index": index_name,
            vector_field: embeddings[i],
            text_field: text,
            "metadata": metadata,
            "_id": _id,
        }
        requests.append(request)
        return_ids.append(_id)
    bulk(client, requests, max_chunk_bytes=max_chunk_bytes)
    client.indices.refresh(index=index_name)
    return return_ids


def _default_approximate_search_query(
    query_vector: List[float],
    k: int = 4,
    vector_field: str = "embedding",
) -> Dict:
    """For Approximate k-NN Search, this is the default query."""
    return {
        "size": k,
        "query": {"knn": {vector_field: {"vector": query_vector, "k": k}}},
    }


def __get_painless_scripting_source(
    space_type: str, vector_field: str = "embedding"
) -> str:
    """For Painless Scripting, it returns the script source based on space type."""
    source_value = f"(1.0 + {space_type}(params.query_value, doc['{vector_field}']))"
    if space_type == "cosineSimilarity":
        return source_value
    else:
        return f"1/{source_value}"


def _default_painless_scripting_query(
    query_vector: List[float],
    k: int = 4,
    space_type: str = "l2Squared",
    pre_filter: Optional[Union[Dict, List]] = None,
    vector_field: str = "embedding",
) -> Dict:
    """For Painless Scripting Search, this is the default query."""
    if not pre_filter:
        pre_filter = MATCH_ALL_QUERY

    source = __get_painless_scripting_source(space_type, vector_field)
    return {
        "size": k,
        "query": {
            "script_score": {
                "query": pre_filter,
                "script": {
                    "source": source,
                    "params": {
                        "field": vector_field,
                        "query_value": query_vector,
                    },
                },
            }
        },
    }


class OpensearchVectorClient:
    """Object encapsulating an Opensearch index that has vector search enabled.

    If the index does not yet exist, it is created during init.
    Therefore, the underlying index is assumed to either:
    1) not exist yet or 2) be created due to previous usage of this class.

    Args:
        endpoint (str): URL (http/https) of elasticsearch endpoint
        index (str): Name of the elasticsearch index
        dim (int): Dimension of the vector
        embedding_field (str): Name of the field in the index to store
            embedding array in.
        text_field (str): Name of the field to grab text from
        method (Optional[dict]): Opensearch "method" JSON obj for configuring
            the KNN index.
            This includes engine, metric, and other config params. Defaults to:
            {"name": "hnsw", "space_type": "l2", "engine": "faiss",
            "parameters": {"ef_construction": 256, "m": 48}}
        **kwargs: Optional arguments passed to the OpenSearch client from opensearch-py.

    """

    def __init__(
        self,
        endpoint: str,
        index: str,
        dim: int,
        embedding_field: str = "embedding",
        text_field: str = "content",
        method: Optional[dict] = None,
        max_chunk_bytes: int = 1 * 1024 * 1024,
        **kwargs: Any,
    ):
        """Init params."""
        if method is None:
            method = {
                "name": "hnsw",
                "space_type": "l2",
                "engine": "nmslib",
                "parameters": {"ef_construction": 256, "m": 48},
            }
        if embedding_field is None:
            embedding_field = "embedding"
        self._embedding_field = embedding_field

        self._endpoint = endpoint
        self._dim = dim
        self._index = index
        self._text_field = text_field
        self._max_chunk_bytes = max_chunk_bytes
        # initialize mapping
        idx_conf = {
            "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 100}},
            "mappings": {
                "properties": {
                    embedding_field: {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": method,
                    },
                }
            },
        }
        self._os_client = _get_opensearch_client(self._endpoint, **kwargs)
        not_found_error = _import_not_found_error()
        try:
            self._os_client.indices.get(index=self._index)
        except not_found_error:
            self._os_client.indices.create(index=self._index, body=idx_conf)
            self._os_client.indices.refresh(index=self._index)

    def index_results(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """Store results in the index."""
        embeddings: List[List[float]] = []
        texts: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []
        for node in nodes:
            ids.append(node.node_id)
            embeddings.append(node.get_embedding())
            texts.append(node.get_content(metadata_mode=MetadataMode.NONE))
            metadatas.append(node_to_metadata_dict(node, remove_text=True))

        return _bulk_ingest_embeddings(
            self._os_client,
            self._index,
            embeddings,
            texts,
            metadatas=metadatas,
            ids=ids,
            vector_field=self._embedding_field,
            text_field=self._text_field,
            mapping=None,
            max_chunk_bytes=self._max_chunk_bytes,
        )

    def delete_doc_id(self, doc_id: str) -> None:
        """Delete a document.

        Args:
            doc_id (str): document id
        """
        self._os_client.delete(index=self._index, id=doc_id)

    def knn(
        self,
        query_embedding: List[float],
        k: int,
        filters: Optional[MetadataFilters] = None,
    ) -> VectorStoreQueryResult:
        """Do knn search.

        If there are no filters do approx-knn search.
        If there are (pre)-filters, do an exhaustive exact knn search using 'painless
            scripting'.

        Note that approximate knn search does not support pre-filtering.

        Args:
            query_embedding: Vector embedding to query.
            k: Maximum number of results.
            filters: Optional filters to apply before the search.
                Supports filter-context queries documented at
                https://opensearch.org/docs/latest/query-dsl/query-filter-context/

        Returns:
            Up to k docs closest to query_embedding
        """
        if filters is None:
            search_query = _default_approximate_search_query(
                query_embedding, k, vector_field=self._embedding_field
            )
        else:
            pre_filter = []
            for f in filters.filters:
                pre_filter.append({f.key: json.loads(str(f.value))})
            # https://opensearch.org/docs/latest/search-plugins/knn/painless-functions/
            search_query = _default_painless_scripting_query(
                query_embedding,
                k,
                space_type="l2Squared",
                pre_filter={"bool": {"filter": pre_filter}},
                vector_field=self._embedding_field,
            )

        res = self._os_client.search(index=self._index, body=search_query)
        nodes = []
        ids = []
        scores = []
        for hit in res["hits"]["hits"]:
            source = hit["_source"]
            node_id = hit["_id"]
            text = source[self._text_field]
            metadata = source.get("metadata", None)

            try:
                node = metadata_dict_to_node(metadata)
                node.text = text
            except Exception:
                # TODO: Legacy support for old nodes
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
            ids.append(node_id)
            nodes.append(node)
            scores.append(hit["_score"])
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)


class OpensearchVectorStore(VectorStore):
    """Elasticsearch/Opensearch vector store.

    Args:
        client (OpensearchVectorClient): Vector index client to use
            for data insertion/querying.
    """

    stores_text: bool = True

    def __init__(
        self,
        client: OpensearchVectorClient,
    ) -> None:
        """Initialize params."""
        self._client = client

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings.

        """
        self._client.index_results(nodes)
        return [result.node_id for result in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        self._client.delete_doc_id(ref_doc_id)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding

        """
        query_embedding = cast(List[float], query.query_embedding)
        return self._client.knn(
            query_embedding, query.similarity_top_k, filters=query.filters
        )
