"""Redis Vector store index.

An index that that is built on top of an existing vector store.
"""
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.readers.redis.utils import (
    TokenEscaper,
    array_to_buffer,
    check_redis_modules_exist,
    convert_bytes,
    get_redis_query,
)
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

_logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from redis.client import Redis as RedisType
    from redis.commands.search.field import VectorField


class RedisVectorStore(VectorStore):
    stores_text = True
    stores_node = True

    tokenizer = TokenEscaper()

    def __init__(
        self,
        index_name: str,
        index_prefix: str = "llama_index",
        index_args: Optional[Dict[str, Any]] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize RedisVectorStore.

        For index arguments that can be passed to RediSearch, see
        https://redis.io/docs/stack/search/reference/vectors/

        The index arguments will depend on the index type chosen. There
        are two available index types
            - FLAT: a flat index that uses brute force search
            - HNSW: a hierarchical navigable small world graph index

        Args:
            index_name (str): Name of the index.
            index_prefix (str): Prefix for the index. Defaults to "llama_index".
            index_args (Dict[str, Any]): Arguments for the index. Defaults to None.
            redis_url (str): URL for the redis instance.
                Defaults to "redis://localhost:6379".
            overwrite (bool): Whether to overwrite the index if it already exists.
                Defaults to False.
            kwargs (Any): Additional arguments to pass to the redis client.

        Raises:
            ValueError: If redis-py is not installed
            ValueError: If RediSearch is not installed

        Examples:
            >>> from llama_index.vector_stores.redis import RedisVectorStore
            >>> # Create a RedisVectorStore
            >>> vector_store = RedisVectorStore(
            >>>     index_name="my_index",
            >>>     index_prefix="gpt_index",
            >>>     index_args={"algorithm": "HNSW", "m": 16, "ef_construction": 200,
                "distance_metric": "cosine"},
            >>>     redis_url="redis://localhost:6379/",
            >>>     overwrite=True)

        """
        try:
            import redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        try:
            # connect to redis from url
            self._redis_client = redis.from_url(redis_url, **kwargs)
            # check if redis has redisearch module installed
            check_redis_modules_exist(self._redis_client)
        except ValueError as e:
            raise ValueError(f"Redis failed to connect: {e}")

        # index identifiers
        self._prefix = index_prefix
        self._index_name = index_name
        self._index_args = index_args if index_args is not None else {}
        self._overwrite = overwrite
        self._vector_field = str(self._index_args.get("vector_field", "vector"))
        self._vector_key = str(self._index_args.get("vector_key", "vector"))

    @property
    def client(self) -> "RedisType":
        """Return the redis client instance"""
        return self._redis_client

    def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        """Add embedding results to the index.

        Args:
            embedding_results (List[NodeWithEmbedding]): List of embedding results to
                add to the index.

        Returns:
            List[str]: List of ids of the documents added to the index.

        Raises:
            ValueError: If the index already exists and overwrite is False.
        """

        # check index exists, call once to avoid calling multiple times
        index_exists = self._index_exists()
        if index_exists and self._overwrite:
            self.delete_index()

        elif index_exists and not self._overwrite:
            raise ValueError("Index already exists and overwrite is False.")

        else:  # index does not exist, create it
            # get vector dim from embedding results if index does not exist
            # as it will be created from the embedding result attributes.
            self._index_args["dims"] = len(embedding_results[0].embedding)
            self._create_index()

        ids = []
        for result in embedding_results:
            # Add extra info and node info to the index
            # cast types to satisfy mypy
            node_info = cast_metadata_types(result.node.node_info)
            extra_info = cast_metadata_types(result.node.extra_info)

            mapping = {
                "id": result.id,
                "doc_id": result.ref_doc_id,
                "text": result.node.get_text(),
                self._vector_key: array_to_buffer(result.embedding),
                **node_info,
                **extra_info,
            }
            ids.append(result.id)
            key = "_".join([self._prefix, str(result.id)])
            self._redis_client.hset(key, mapping=mapping)  # type: ignore

        _logger.info(f"Added {len(ids)} documents to index {self._index_name}")
        return ids

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a specific document from the index by doc_id

        Args:
            doc_id (str): The doc_id of the document to delete.
            delete_kwargs (Any): Additional arguments to pass to the delete method.

        """
        # use tokenizer to escape dashes in query
        query_str = "@doc_id:{%s}" % self.tokenizer.escape(doc_id)
        # find all documents that match a doc_id
        results = self._redis_client.ft(self._index_name).search(query_str)

        for doc in results.docs:
            self._redis_client.delete(doc.id)
        _logger.info(
            f"Deleted {len(results.docs)} documents from index {self._index_name}"
        )

    def delete_index(self) -> None:
        """Delete the index and all documents."""
        _logger.info(f"Deleting index {self._index_name}")
        self._redis_client.ft(self._index_name).dropindex(delete_documents=True)

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Query the index.

        Args:
            query (VectorStoreQuery): query object

        Returns:
            VectorStoreQueryResult: query result
        """
        from redis.exceptions import ResponseError as RedisResponseError

        return_fields = ["id", "doc_id", "text", self._vector_key, "vector_score"]

        redis_query = get_redis_query(
            return_fields=return_fields,
            top_k=query.similarity_top_k,
            vector_field=self._vector_field,
        )
        if not query.query_embedding:
            raise ValueError("Query embedding is required for querying.")

        query_params = {
            "vector": array_to_buffer(query.query_embedding),
        }
        _logger.info(f"Querying index {self._index_name}")

        try:
            results = self._redis_client.ft(self._index_name).search(
                redis_query, query_params=query_params  # type: ignore
            )
        except RedisResponseError as e:
            _logger.error(f"Error querying index {self._index_name}: {e}")
            raise e

        ids = []
        nodes = []
        scores = []
        for doc in results.docs:
            node = Node(
                text=doc.text,
                doc_id=doc.id,
                embedding=None,
                relationships={DocumentRelationship.SOURCE: doc.doc_id},
            )
            ids.append(doc.id)
            nodes.append(node)
            scores.append(1 - float(doc.vector_score))
        _logger.info(f"Found {len(nodes)} results for query with id {ids}")

        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)

    def persist(self, persist_path: str, in_background: bool = True) -> None:
        """Persist the vector store to disk.

        Args:
            persist_path (str): Path to persist the vector store to. (doesn't apply)
            in_background (bool, optional): Persist in background. Defaults to True.
        """
        if in_background:
            _logger.info("Saving index to disk in background")
            self._redis_client.bgsave()
        else:
            _logger.info("Saving index to disk")
            self._redis_client.save()

    def _create_index(self) -> None:
        # should never be called outside class and hence should not raise importerror
        from redis.commands.search.field import TagField, TextField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType

        # Create Index
        default_fields = [
            TextField("text", weight=1.0),
            TagField("doc_id", sortable=False),
            TagField("id", sortable=False),
        ]
        # add vector field to list of index fields. Create lazily to allow user
        # to specify index and search attributes in creation.
        fields = default_fields + [
            self._create_vector_field(self._vector_field, **self._index_args)
        ]

        _logger.info(f"Creating index {self._index_name}")
        self._redis_client.ft(self._index_name).create_index(
            fields=fields,
            definition=IndexDefinition(
                prefix=[self._prefix], index_type=IndexType.HASH
            ),  # TODO support JSON
        )

    def _index_exists(self) -> bool:
        # use FT._LIST to check if index exists
        indices = convert_bytes(self._redis_client.execute_command("FT._LIST"))
        return self._index_name in indices

    def _create_vector_field(
        self,
        name: str,
        dims: int = 1536,
        algorithm: str = "FLAT",
        datatype: str = "FLOAT32",
        distance_metric: str = "COSINE",
        initial_cap: int = 20000,
        block_size: int = 1000,
        m: int = 16,
        ef_construction: int = 200,
        ef_runtime: int = 10,
        epsilon: float = 0.8,
        **kwargs: Any,
    ) -> "VectorField":
        """Create a RediSearch VectorField.

        Args:
            name (str): The name of the field.
            algorithm (str): The algorithm used to index the vector.
            dims (int): The dimensionality of the vector.
            datatype (str): The type of the vector. default: FLOAT32
            distance_metric (str): The distance metric used to compare vectors.
            initial_cap (int): The initial capacity of the index.
            block_size (int): The block size of the index.
            m (int): The number of outgoing edges in the HNSW graph.
            ef_construction (int): Number of maximum allowed potential outgoing edges
                            candidates for each node in the graph,
                            during the graph building.
            ef_runtime (int): The umber of maximum top candidates to hold during the
                KNN search

        returns:
            A RediSearch VectorField.
        """
        from redis.commands.search.field import VectorField

        if algorithm.upper() == "HNSW":
            return VectorField(
                name,
                "HNSW",
                {
                    "TYPE": datatype.upper(),
                    "DIM": dims,
                    "DISTANCE_METRIC": distance_metric.upper(),
                    "INITIAL_CAP": initial_cap,
                    "M": m,
                    "EF_CONSTRUCTION": ef_construction,
                    "EF_RUNTIME": ef_runtime,
                    "EPSILON": epsilon,
                },
            )
        else:
            return VectorField(
                name,
                "FLAT",
                {
                    "TYPE": datatype.upper(),
                    "DIM": dims,
                    "DISTANCE_METRIC": distance_metric.upper(),
                    "INITIAL_CAP": initial_cap,
                    "BLOCK_SIZE": block_size,
                },
            )


def cast_metadata_types(mapping: Optional[Dict[str, Any]]) -> Dict[str, str]:
    metadata = {}
    if mapping:
        for key, value in mapping.items():
            metadata[str(key)] = str(value)
    return metadata
