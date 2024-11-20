"""Redis Vector store index.

An index that is built on top of an existing vector store.
"""

import logging
from typing import Any, Dict, List, Optional

import fsspec
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    MetadataFilter,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.vector_stores.redis.schema import (
    NODE_ID_FIELD_NAME,
    NODE_CONTENT_FIELD_NAME,
    DOC_ID_FIELD_NAME,
    TEXT_FIELD_NAME,
    VECTOR_FIELD_NAME,
    RedisVectorStoreSchema,
)
from llama_index.vector_stores.redis.utils import REDIS_LLAMA_FIELD_SPEC

from redis import Redis
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery, FilterQuery, CountQuery
from redisvl.query.filter import Tag, FilterExpression
from redisvl.schema.fields import BaseField
from redisvl.redis.utils import array_to_buffer


logger = logging.getLogger(__name__)


class RedisVectorStore(BasePydanticVectorStore):
    """RedisVectorStore.

    The RedisVectorStore takes a user-defined schema object and a Redis connection
    client or URL string. The schema is optional, but useful for:
    - Defining a custom index name, key prefix, and key separator.
    - Defining *additional* metadata fields to use as query filters.
    - Setting custom specifications on fields to improve search quality, e.g
    which vector index algorithm to use.

    Other Notes:
    - All embeddings and docs are stored in Redis. During query time, the index
    uses Redis to query for the top k most similar nodes.
    - Redis & LlamaIndex expect at least 4 *required* fields for any schema, default or custom,
    `id`, `doc_id`, `text`, `vector`.

    Args:
        schema (IndexSchema, optional): Redis index schema object.
        redis_client (Redis, optional): Redis client connection.
        redis_url (str, optional): Redis server URL.
            Defaults to "redis://localhost:6379".
        overwrite (bool, optional): Whether to overwrite the index if it already exists.
            Defaults to False.

    Raises:
        ValueError: If your Redis server does not have search or JSON enabled.
        ValueError: If a Redis connection failed to be established.
        ValueError: If an invalid schema is provided.

    Example:
        from redisvl.schema import IndexSchema
        from llama_index.vector_stores.redis import RedisVectorStore

        # Use default schema
        rds = RedisVectorStore(redis_url="redis://localhost:6379")

        # Use custom schema from dict
        schema = IndexSchema.from_dict({
            "index": {"name": "my-index", "prefix": "docs"},
            "fields": [
                {"name": "id", "type": "tag"},
                {"name": "doc_id", "type": "tag},
                {"name": "text", "type": "text"},
                {"name": "vector", "type": "vector", "attrs": {"dims": 1536, "algorithm": "flat"}}
            ]
        })
        vector_store = RedisVectorStore(
            schema=schema,
            redis_url="redis://localhost:6379"
        )
    """

    stores_text: bool = True
    stores_node: bool = True
    flat_metadata: bool = False

    _index: SearchIndex = PrivateAttr()
    _overwrite: bool = PrivateAttr()
    _return_fields: List[str] = PrivateAttr()

    def __init__(
        self,
        schema: Optional[IndexSchema] = None,
        redis_client: Optional[Redis] = None,
        redis_url: Optional[str] = None,
        overwrite: bool = False,
        return_fields: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # check for indicators of old schema
        self._flag_old_kwargs(**kwargs)

        # Setup schema
        if not schema:
            logger.info("Using default RedisVectorStore schema.")
            schema = RedisVectorStoreSchema()

        self._validate_schema(schema)
        self._return_fields = return_fields or [
            NODE_ID_FIELD_NAME,
            DOC_ID_FIELD_NAME,
            TEXT_FIELD_NAME,
            NODE_CONTENT_FIELD_NAME,
        ]
        self._index = SearchIndex(schema=schema)
        self._overwrite = overwrite

        # Establish redis connection
        if redis_client:
            self._index.set_client(redis_client)
        elif redis_url:
            self._index.connect(redis_url)
        else:
            raise ValueError(
                "Failed to connect to Redis. Must provide a valid redis client or url"
            )

        # Create index
        self.create_index()

    def _flag_old_kwargs(self, **kwargs):
        old_kwargs = [
            "index_name",
            "index_prefix",
            "prefix_ending",
            "index_args",
            "metadata_fields",
        ]
        for kwarg in old_kwargs:
            if kwarg in kwargs:
                raise ValueError(
                    f"Deprecated kwarg, {kwarg}, found upon initialization. "
                    "RedisVectorStore now requires an IndexSchema object. "
                    "See the documentation for a complete example: https://docs.llamaindex.ai/en/stable/examples/vector_stores/RedisIndexDemo/"
                )

    def _validate_schema(self, schema: IndexSchema) -> str:
        base_schema = RedisVectorStoreSchema()
        for name, field in base_schema.fields.items():
            if (name not in schema.fields) or (
                not schema.fields[name].type == field.type
            ):
                raise ValueError(
                    f"Required field {name} must be present in the index "
                    f"and of type {schema.fields[name].type}"
                )

    @property
    def client(self) -> "Redis":
        """Return the redis client instance."""
        return self._index.client

    @property
    def index_name(self) -> str:
        """Return the name of the index based on the schema."""
        return self._index.name

    @property
    def schema(self) -> IndexSchema:
        """Return the index schema."""
        return self._index.schema

    def set_return_fields(self, return_fields: List[str]) -> None:
        """Update the return fields for the query response."""
        self._return_fields = return_fields

    def index_exists(self) -> bool:
        """Check whether the index exists in Redis.

        Returns:
            bool: True or False.
        """
        return self._index.exists()

    def create_index(self, overwrite: Optional[bool] = None) -> None:
        """Create an index in Redis."""
        if overwrite is None:
            overwrite = self._overwrite
        # Create index honoring overwrite policy
        if overwrite:
            self._index.create(overwrite=True, drop=True)
        else:
            self._index.create()

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to the index.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings

        Returns:
            List[str]: List of ids of the documents added to the index.

        Raises:
            ValueError: If the index already exists and overwrite is False.
        """
        # Check to see if empty document list was passed
        if len(nodes) == 0:
            return []

        # Now check for the scenario where user is trying to index embeddings that don't align with schema
        embedding_len = len(nodes[0].get_embedding())
        expected_dims = self._index.schema.fields[VECTOR_FIELD_NAME].attrs.dims
        if expected_dims != embedding_len:
            raise ValueError(
                f"Attempting to index embeddings of dim {embedding_len} "
                f"which doesn't match the index schema expectation of {expected_dims}. "
                "Please review the Redis integration example to learn how to customize schema. "
                ""
            )

        data: List[Dict[str, Any]] = []
        for node in nodes:
            embedding = node.get_embedding()
            record = {
                NODE_ID_FIELD_NAME: node.node_id,
                DOC_ID_FIELD_NAME: node.ref_doc_id,
                TEXT_FIELD_NAME: node.get_content(metadata_mode=MetadataMode.NONE),
                VECTOR_FIELD_NAME: array_to_buffer(embedding, dtype="FLOAT32"),
            }
            # parse and append metadata
            additional_metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )
            data.append({**record, **additional_metadata})

        # Load nodes to Redis
        keys = self._index.load(data, id_field=NODE_ID_FIELD_NAME, **add_kwargs)
        logger.info(f"Added {len(keys)} documents to index {self._index.name}")
        return [
            key.strip(self._index.prefix + self._index.key_separator) for key in keys
        ]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        # build a filter to target specific docs by doc ID
        doc_filter = Tag(DOC_ID_FIELD_NAME) == ref_doc_id
        total = self._index.query(CountQuery(doc_filter))
        delete_query = FilterQuery(
            return_fields=[NODE_ID_FIELD_NAME],
            filter_expression=doc_filter,
            num_results=total,
        )
        # fetch docs to delete and flush them
        docs_to_delete = self._index.search(delete_query.query, delete_query.params)
        with self._index.client.pipeline(transaction=False) as pipe:
            for doc in docs_to_delete.docs:
                pipe.delete(doc.id)
            res = pipe.execute()

        logger.info(
            f"Deleted {len(docs_to_delete.docs)} documents from index {self._index.name}"
        )

    def delete_index(self) -> None:
        """Delete the index and all documents."""
        logger.info(f"Deleting index {self._index.name}")
        self._index.delete(drop=True)

    @staticmethod
    def _to_redis_filter(field: BaseField, filter: MetadataFilter) -> FilterExpression:
        """
        Translate a standard metadata filter to a Redis specific filter expression.

        Args:
            field (BaseField): The field to be filtered on, must have a type attribute.
            filter (MetadataFilter): The filter to apply, must have operator and value attributes.

        Returns:
            FilterExpression: A Redis-specific filter expression constructed from the input.

        Raises:
            ValueError: If the field type is unsupported or if the operator is not supported for the field type.
        """
        # Check for unsupported field type
        if field.type not in REDIS_LLAMA_FIELD_SPEC:
            raise ValueError(f"Unsupported field type {field.type} for {field.name}")

        field_info = REDIS_LLAMA_FIELD_SPEC[field.type]

        # Check for unsupported operator
        if filter.operator not in field_info["operators"]:
            raise ValueError(
                f"Filter operator {filter.operator} not supported for {field.name} of type {field.type}"
            )

        # Create field instance and apply the operator function
        field_instance = field_info["class"](field.name)
        return field_info["operators"][filter.operator](field_instance, filter.value)

    def _create_redis_filter_expression(
        self, metadata_filters: MetadataFilters
    ) -> FilterExpression:
        """
        Generate a Redis Filter Expression as a combination of metadata filters.

        Args:
            metadata_filters (MetadataFilters): List of metadata filters to use.

        Returns:
            FilterExpression: A Redis filter expression.
        """
        filter_expression = FilterExpression("*")
        if metadata_filters:
            if metadata_filters.filters:
                for filter in metadata_filters.filters:
                    # Handle nested MetadataFilters recursively
                    if isinstance(filter, MetadataFilters):
                        redis_filter = self._create_redis_filter_expression(filter)
                    else:
                        # Index must be created with the metadata field in the index schema
                        field = self._index.schema.fields.get(filter.key)
                        if not field:
                            logger.warning(
                                f"{filter.key} field was not included as part of the index schema, and thus cannot be used as a filter condition."
                            )
                            continue
                        # Extract redis filter
                        redis_filter = self._to_redis_filter(field, filter)

                    # Combine with conditional
                    if metadata_filters.condition == "and":
                        filter_expression = filter_expression & redis_filter
                    else:
                        filter_expression = filter_expression | redis_filter
        return filter_expression

    def _to_redis_query(self, query: VectorStoreQuery) -> VectorQuery:
        """Creates a RedisQuery from a VectorStoreQuery."""
        filter_expression = self._create_redis_filter_expression(query.filters)
        return_fields = self._return_fields.copy()
        return VectorQuery(
            vector=query.query_embedding,
            vector_field_name=VECTOR_FIELD_NAME,
            num_results=query.similarity_top_k,
            filter_expression=filter_expression,
            return_fields=return_fields,
        )

    def _extract_node_and_score(self, doc, redis_query: VectorQuery):
        """Extracts a node and its score from a document."""
        try:
            node = metadata_dict_to_node(
                {NODE_CONTENT_FIELD_NAME: doc[NODE_CONTENT_FIELD_NAME]}
            )
            node.text = doc[TEXT_FIELD_NAME]
        except Exception:
            # Handle legacy metadata format
            node = TextNode(
                text=doc[TEXT_FIELD_NAME],
                id_=doc[NODE_ID_FIELD_NAME],
                embedding=None,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(
                        node_id=doc[DOC_ID_FIELD_NAME]
                    )
                },
            )
        score = 1 - float(doc[redis_query.DISTANCE_ID])
        return node, score

    def _process_query_results(
        self, results, redis_query: VectorQuery
    ) -> VectorStoreQueryResult:
        """Processes query results and returns a VectorStoreQueryResult."""
        ids, nodes, scores = [], [], []
        for doc in results:
            node, score = self._extract_node_and_score(doc, redis_query)
            ids.append(doc[NODE_ID_FIELD_NAME])
            nodes.append(node)
            scores.append(score)
        logger.info(f"Found {len(nodes)} results for query with id {ids}")
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query the index.

        Args:
            query (VectorStoreQuery): query object

        Returns:
            VectorStoreQueryResult: query result

        Raises:
            ValueError: If query.query_embedding is None.
            redis.exceptions.RedisError: If there is an error querying the index.
            redis.exceptions.TimeoutError: If there is a timeout querying the index.
        """
        if not query.query_embedding:
            raise ValueError("Query embedding is required for querying.")

        redis_query = self._to_redis_query(query)
        logger.info(f"Querying index {self._index.name} with query {redis_query!s}")

        try:
            results = self._index.query(redis_query)
        except RedisTimeoutError as e:
            logger.error(f"Query timed out on {self._index.name}: {e}")
            raise
        except RedisError as e:
            logger.error(f"Error querying {self._index.name}: {e}")
            raise

        return self._process_query_results(results, redis_query)

    def persist(
        self,
        persist_path: Optional[str] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        in_background: bool = True,
    ) -> None:
        """Persist the vector store to disk.

        For Redis, more notes here: https://redis.io/docs/management/persistence/

        Args:
            persist_path (str): Path to persist the vector store to. (doesn't apply)
            in_background (bool, optional): Persist in background. Defaults to True.
            fs (fsspec.AbstractFileSystem, optional): Filesystem to persist to.
                (doesn't apply)

        Raises:
            redis.exceptions.RedisError: If there is an error
                                         persisting the index to disk.
        """
        try:
            if in_background:
                logger.info("Saving index to disk in background")
                self._index.client.bgsave()
            else:
                logger.info("Saving index to disk")
                self._index.client.save()

        except RedisError as e:
            logger.error(f"Error saving index to disk: {e}")
            raise
