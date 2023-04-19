"""Redis Vector store index.

An index that that is built on top of an existing vector store.
"""
import ast
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from redis.commands.search.field import Field, TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from gpt_index.data_structs.node_v2 import DocumentRelationship, Node
from gpt_index.readers.redis import get_redis_query
from gpt_index.vector_stores.types import (NodeEmbeddingResult, VectorStore,
                                           VectorStoreQuery,
                                           VectorStoreQueryMode,
                                           VectorStoreQueryResult)

_logger = logging.getLogger(__name__)

class RedisVectorStore(VectorStore):

    stores_text = True
    default_fields = [
        TextField("text", weight=1.0),
        TagField("doc_id", sortable=False),
        TagField("id", sortable=False)
    ]

    def __init__(
        self,
        index_name: Optional[str],
        index_prefix: Optional[str] = "gpt_index",
        redis_client: Optional[Any] = None,
        vector_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RedisVectorStore."""
        if redis_client is None:
            raise ValueError("Missing Redis client!")
        self._redis_client = redis_client
        self._prefix = index_prefix
        self._index_name = index_name

        # Create the index fields with the default fields if user doesn't specify
        self._fields = kwargs.get("fields", self.default_fields)

        # User-specified vector index attributes
        self._vector_args = vector_args or {}


    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VectorStore":
        """Return RedisVectorStore instance from a dictionary of config params."""
        import redis

        client = redis.Redis(
            host=config_dict["host"],
            port=config_dict["port"],
            username=config_dict.get("username"),
            password=config_dict.get("password"),
        )
        # TODO Check if index name exists
        return cls(
            index_name=config_dict["index_name"],
            index_prefix=config_dict.get("prefix", "gpt_index"),
            redis_client=client,
        )

    @property
    def client(self) -> Any:
        """Return the redis client instance"""
        return self._redis_client

    @property
    def config_dict(self) -> Dict[str, Any]:
        """Return the config dictionary."""
        # TODO debate whether to return the password
        return {
            "host": self._redis_client.connection_pool.connection_kwargs["host"],
            "port": self._redis_client.connection_pool.connection_kwargs["port"],
            "username": self._redis_client.connection_pool.connection_kwargs.get("username"),
            "password": self._redis_client.connection_pool.connection_kwargs.get("password"),
            "index_name": self._index_name,
            "prefix": self._prefix,
        }

    def add(self, embedding_results: List[NodeEmbeddingResult]) -> List[str]:
        if not self._index_exists():
            self._update_schema()
            _logger.info(f"Creating index {self._index_name} with fields {self._fields}")
            self._create_index()

        ids = []
        for result in embedding_results:
            ids.append(result.id)
            mapping = {
                "id": result.id,
                "doc_id": result.doc_id,
                "text": result.node.text,
                "vector": np.array(result.embedding).astype(np.float32).tobytes(),
                **result.node.node_info
            }
            key = self._prefix + "_" + str(result.id)
            self._redis_client.hset(key, mapping=mapping)
        _logger.info(f"Added {len(ids)} documents to index {self._index_name}")
        return ids


    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete doc."""
        self._redis_client.delete(self._prefix + doc_id)


    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Query the index.

        Args:
            query (VectorStoreQuery): query object

        Returns:
            VectorStoreQueryResult: query result
        """
        return_fields = ["id", "doc_id", "text"]
        redis_query = get_redis_query(
            return_fields=return_fields,
            top_k=query.similarity_top_k,
        )
        query_params = {"vector": np.array(query.query_embedding).astype(np.float32).tobytes()}
        _logger.info(f"Querying index {self._index_name}")
        results = self._redis_client.ft(self._index_name).search(redis_query,
                                                                 query_params=query_params)

        ids = []
        nodes = []
        for doc in results.docs:
            ids.append(doc.id)
            node = Node(
                text=doc.text,
                doc_id=doc.doc_id,
                embedding=None, # TODO Return vectors as well (problems with bytes to float list conversion)
            )
            nodes.append(node)
        _logger.info(f"Found {len(nodes)} results for query with id {ids}")
        return VectorStoreQueryResult(nodes=nodes, ids=ids)


    def _create_index(self) -> None:
        # Create Index
        self._redis_client.ft(self._index_name).create_index(
            self._fields,
            definition=IndexDefinition(prefix=[self._prefix], index_type=IndexType.HASH) # TODO support JSON
        )

    def _update_schema(self):
        # update the fields with the vector field
        vector_field = self._create_vector_field(**self._vector_args)
        self._fields.append(vector_field)

    def _index_exists(self) -> bool:
        # use FT._LIST to check if index exists

        indices = convert_bytes(self._redis_client.execute_command("FT._LIST"))
        return self._index_name in indices


    def _create_vector_field(
        self,
        name: str = "vector",
        dims: int = "1536",
        algorithm: str = "FLAT",
        datatype: str = "FLOAT32",
        distance_metric: str = "COSINE",
        initial_cap: int = 20000,
        block_size: int = 1000,
        m: int = 16,
        ef_construction: int = 200,
        ef_runtime: int = 10,
        epsilon: float = 0.8,
        **kwargs: Any
    ):
        """Create a RediSearch VectorField.

        Args:
        name: The name of the field.
        algorithm: The algorithm used to index the vector.
        dims: The dimensionality of the vector.
        datatype: The type of the vector. default: FLOAT32
        distance_metric: The distance metric used to compare vectors.
        initial_cap: The initial capacity of the index.
        block_size: The block size of the index.
        m: The number of outgoing edges in the HNSW graph.
        ef_construction: Number of maximum allowed potential outgoing edges
                        candidates for each node in the graph, during the graph building.
        ef_runtime: The umber of maximum top candidates to hold during the KNN search

        returns:
        A RediSearch VectorField.
        """
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


def convert_bytes(data):
    if isinstance(data, bytes):  return data.decode('ascii')
    if isinstance(data, dict):   return dict(map(convert_bytes, data.items()))
    if isinstance(data, list):   return list(map(convert_bytes, data))
    if isinstance(data, tuple):  return map(convert_bytes, data)
    return data
