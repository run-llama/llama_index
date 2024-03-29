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
from llama_index.vector_stores.redis.schema import RedisVectorStoreSchema
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
    stores_text = True
    stores_node = True
    flat_metadata = False

    _overwrite: bool = PrivateAttr()
    _return_fields: List[str] = PrivateAttr()
    _vector_field_name: str = PrivateAttr()
    _node_id_field_name: str = "id"
    _doc_id_field_name: str = "doc_id"
    _text_field_name: str = "text"

    def __init__(
        self,
        schema: Optional[IndexSchema] = None,
        redis_client: Optional[Redis] = None,
        redis_url: Optional[str] = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize RedisVectorStore.

        Args:
            schema (IndexSchema, optional): Redis index schema object.
            redis_client (Redis, optional): Redis client connection.
            redis_url (str, optional): Redis server URL.
                Defaults to "redis://localhost:6379".
            overwrite (bool, optional): Whether to overwrite the index if it already exists.
                Defaults to False.

        Raises:
            ValueError: If redis-py is not installed
            ValueError: If RediSearch is not installed

        # TODO -- update example
        """
        # Setup schema
        if not schema:
            logger.info("Using default RedisVectorStore schema.")
            schema = RedisVectorStoreSchema()

        self._validate_schema(schema)

        self._vector_field_name = next(
            [name for name, field in schema.fields.items() if field.type == "vector"]
        )
        self._return_fields = [
            self._node_id_field_name,
            self._doc_id_field_name,
            self._text_field_name,
            self._text_field_name,
            VectorQuery.DISTANCE_ID,
            "_node_content",
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

        super().__init__()

    def _validate_schema(self, schema: IndexSchema):
        # TODO
        pass

    @property
    def client(self) -> "Redis":
        """Return the redis client instance."""
        return self._index.client

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to the index.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings

        Returns:
            List[str]: List of ids of the documents added to the index.

        Raises:
            ValueError: If the index already exists and overwrite is False.
        """
        # check to see if empty document list was passed
        if len(nodes) == 0:
            return []

        # set vector dim for creation if index doesn't exist
        # self._index_args["dims"] = len(nodes[0].get_embedding())
        # TODO discuss with logan -- this is risky... updating schema based on provided embeddings...

        if self._index.exists():
            if self._overwrite:
                self._index.create(overwrite=True, drop=True)
        else:
            self._index.create()

        def preprocess_node(node: BaseNode) -> Dict[str, Any]:
            record = {
                self._node_id_field_name: node.node_id,
                self._doc_id_field_name: node.ref_doc_id,
                self._text_field_name: node.get_content(
                    metadata_mode=MetadataMode.NONE
                ),
                self._vector_field_name: array_to_buffer(node.get_embedding()),
            }
            # parse and append metadata
            additional_metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )
            record.update(additional_metadata)
            return record

        keys = self._index.load(
            data=nodes, preprocess=preprocess_node, id_field="id", **add_kwargs
        )
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
        doc_filter = Tag(self._doc_id_field_name) == ref_doc_id
        total = self._index.query(CountQuery(doc_filter))
        delete_query = FilterQuery(
            return_fields=[self._node_id_field_name],
            filter_expression=doc_filter,
            num_results=total,
        )
        # fetch docs to delete and flush them
        docs_to_delete = self._index.search(delete_query.query, delete_query.params)
        with self._index.client.pipeline(transaction=False) as pipe:
            for doc in docs_to_delete:
                pipe.delete(doc.id)
            res = pipe.execute()

        logger.info(
            f"Deleted {len(docs_to_delete)} documents from index {self._index.name}"
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
        if metadata_filters.filters:
            for filter in metadata_filters.filters:
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
        return VectorQuery(
            vector=query.query_embedding,
            vector_field_name=self._vector_field_name,
            return_fields=self._return_fields,
            num_results=query.similarity_top_k,
            filter_expression=filter_expression,
        )

    def _extract_node_and_score(self, doc, redis_query: VectorQuery):
        """Extracts a node and its score from a document."""
        try:
            # TODO investigate this a bit more
            node = metadata_dict_to_node({"_node_content": doc["_node_content"]})
            node.text = doc[self._text_field_name]
        except Exception:
            # Handle legacy metadata format
            node = TextNode(
                text=doc[self._text_field_name],
                id_=doc[self._id_field_name],
                embedding=None,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(
                        node_id=doc[self._doc_id_field_name]
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
            ids.append(doc[self._node_id_field_name])
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
            ValueError: If no documents are found when querying the index.
        """
        if not query.query_embedding:
            raise ValueError("Query embedding is required for querying.")

        redis_query = self._to_redis_query(query)
        logger.info(
            f"Querying index {self._index.name} with filters {redis_query.filters}"
        )

        try:
            results = self._index.query(redis_query)
        except RedisTimeoutError as e:
            logger.error(f"Query timed out on {self._index.name}: {e}")
            raise
        except RedisError as e:
            logger.error(f"Error querying {self._index.name}: {e}")
            raise

        if not results:
            raise ValueError(
                f"No docs found on index '{self._index.name}' with "
                f"prefix '{self._index.prefix}' and filters '{redis_query.get_filter()}'. "
                "* Did you originally create the index with a different prefix? "
                "* Did you index your metadata fields when you created the index?"
            )

        return self._process_query_results(results, redis_query)

    def persist(
        self,
        persist_path: str,
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
