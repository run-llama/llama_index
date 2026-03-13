"""
Valkey Vector store.
"""

import json
import logging
import struct
import traceback
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from glide_sync import Batch as SyncBatch
from glide import ClusterBatch, Batch
from glide import (
    GlideClient,
    GlideClientConfiguration,
    GlideClusterClient,
    ft,
)
from glide_shared import (
    NodeAddress,
)
from glide_shared.commands.server_modules.ft_options.ft_create_options import (
    FtCreateOptions,
)
from glide_shared.commands.server_modules.ft_options.ft_search_options import (
    FtSearchLimit,
    FtSearchOptions,
    ReturnField,
)
from glide_sync import (
    GlideClient as SyncGlideClient,
    GlideClientConfiguration as SyncGlideClientConfiguration,
)
from glide_sync import (
    GlideClusterClient as SyncGlideClusterClient,
)
from glide_sync import ft as sync_ft
from llama_index.vector_stores.valkey.exceptions import ValkeyVectorStoreError
from llama_index.vector_stores.valkey.schema import (
    DOC_ID_FIELD_NAME,
    NODE_CONTENT_FIELD_NAME,
    NODE_ID_FIELD_NAME,
    TEXT_FIELD_NAME,
    VECTOR_FIELD_NAME,
    ValkeyVectorStoreSchema,
    DEFAULT_EMBEDDING_DIM,
)

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
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
    FilterOperator,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)
NO_DOCS = "No docs found on index"


def array_to_buffer(array: List[float], dtype: str = "FLOAT32") -> bytes:
    """Convert array to binary buffer for Valkey."""
    if dtype == "FLOAT32":
        return struct.pack(f"{len(array)}f", *array)
    raise ValueError(f"Unsupported dtype: {dtype}")


class ValkeyVectorStore(BasePydanticVectorStore):
    """
    ValkeyVectorStore.

    The ValkeyVectorStore takes a user-defined schema object and a Valkey connection
    client or URL string. The schema is optional, but useful for:
    - Defining a custom index name, key prefix, and key separator.
    - Defining *additional* metadata fields to use as query filters.
    - Setting custom specifications on fields to improve search quality, e.g
    which vector index algorithm to use.

    Other Notes:
    - All embeddings and docs are stored in Valkey. During query time, the index
    uses Valkey to query for the top k most similar nodes.
    - Valkey & LlamaIndex expect at least 4 *required* fields for any schema, default or custom,
    `id`, `doc_id`, `text`, `vector`.

    Args:
        schema (ValkeyVectorStoreSchema, optional): Valkey index schema object.
        valkey_client (Any, optional): Valkey client connection.
        valkey_url (str, optional): Valkey server URL.
            Defaults to "valkey://localhost:6379".
        overwrite (bool, optional): Whether to overwrite the index if it already exists.
            Defaults to False.

    Raises:
        ValueError: If your Valkey server does not have search or JSON enabled.
        ValueError: If a Valkey connection failed to be established.
        ValueError: If an invalid schema is provided.

    Example:
        from llama_index.vector_stores.valkey import ValkeyVectorStore

        # Use default schema
        vds = ValkeyVectorStore(valkey_url="valkey://localhost:6379")

        # Use custom schema
        from llama_index.vector_stores.valkey.schema import ValkeyVectorStoreSchema

        schema = ValkeyVectorStoreSchema()
        schema.add_fields([{"name": "category", "type": "tag"}])

        vector_store = ValkeyVectorStore(
            schema=schema,
            valkey_url="valkey://localhost:6379"
        )

    """

    stores_text: bool = True
    stores_node: bool = True
    flat_metadata: bool = False
    created_async_index: bool = False
    legacy_filters: bool = False

    _valkey_client: Any = PrivateAttr()
    _valkey_client_async: Any = PrivateAttr()
    _pending_sync_config: Any = PrivateAttr()
    _pending_async_config: Any = PrivateAttr()
    _prefix: str = PrivateAttr()
    _index_args: Dict[str, Any] = PrivateAttr()
    _metadata_fields: List[str] = PrivateAttr()
    _overwrite: bool = PrivateAttr()
    _return_fields: List[str] = PrivateAttr()
    _schema: ValkeyVectorStoreSchema = PrivateAttr()
    _is_cluster: bool = PrivateAttr()

    def __init__(
        self,
        schema: Optional[ValkeyVectorStoreSchema] = None,
        valkey_client: SyncGlideClient | SyncGlideClusterClient | None = None,
        valkey_client_async: GlideClient | GlideClusterClient | None = None,
        valkey_url: Optional[str] = None,
        overwrite: bool = False,
        return_fields: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Setup schema
        if not schema:
            logger.info("Using default ValkeyVectorStore schema.")
            schema = ValkeyVectorStoreSchema()

        self._schema = schema
        self._return_fields = return_fields or [
            NODE_ID_FIELD_NAME,
            DOC_ID_FIELD_NAME,
            TEXT_FIELD_NAME,
            NODE_CONTENT_FIELD_NAME,
        ]
        self._overwrite = overwrite

        # Setup clients
        if valkey_url:
            # Parse URL using urllib
            parsed = urlparse(valkey_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 6379

            # Store config for lazy sync client creation
            if not valkey_client:
                sync_config = SyncGlideClientConfiguration(
                    addresses=[NodeAddress(host, port)]
                )
                self._pending_sync_config = sync_config
                self._valkey_client = None
            else:
                self._valkey_client = valkey_client
                self._pending_sync_config = None

            # Store config for lazy async client creation
            if not valkey_client_async:
                async_config = GlideClientConfiguration(
                    addresses=[NodeAddress(host, port)]
                )
                self._pending_async_config = async_config
                self._valkey_client_async = None
            else:
                self._valkey_client_async = valkey_client_async
                self._pending_async_config = None

            self._is_cluster = False
        else:
            self._pending_sync_config = None
            self._pending_async_config = None
            self._valkey_client = valkey_client
            self._valkey_client_async = valkey_client_async
            self._is_cluster = (
                isinstance(valkey_client, GlideClusterClient)
                if valkey_client
                else False
            )

        if (
            not self._valkey_client
            and not self._valkey_client_async
            and not self._pending_sync_config
            and not self._pending_async_config
        ):
            raise ValkeyVectorStoreError(
                "Either valkey_client, valkey_url, or valkey_client_async need to be defined"
            )

    @property
    def client(
        self,
    ) -> SyncGlideClient | SyncGlideClusterClient:
        """Return the sync valkey client instance."""
        if not self._valkey_client:
            raise ValkeyVectorStoreError("No sync client available")
        return self._valkey_client

    @property
    def aclient(
        self,
    ) -> GlideClient | GlideClusterClient:
        """Return the async valkey client instance."""
        if not self._valkey_client_async:
            raise ValkeyVectorStoreError(
                "No async client available. Use async methods or provide valkey_client_async."
            )
        return self._valkey_client_async

    def _ensure_sync_client(self) -> None:
        """
        Ensure sync client is available, creating it lazily if needed.

        Raises:
            ValkeyVectorStoreError: If sync client cannot be created.

        """
        if self._valkey_client:
            # Already have sync client
            return

        if self._pending_sync_config:
            # Create sync client from pending config
            try:
                self._valkey_client = SyncGlideClient.create(self._pending_sync_config)
                logger.info("Created sync client from URL configuration")
                # Clear pending config after creating client
                self._pending_sync_config = None
            except Exception as e:
                raise ValkeyVectorStoreError(
                    f"Failed to create sync client: {e}"
                ) from e
        else:
            # No way to create sync client
            raise ValkeyVectorStoreError(
                "No sync client available. Either provide valkey_client_async, "
                "valkey_url, or both valkey_client and valkey_client_async."
            )

    async def _ensure_async_client(self) -> None:
        """
        Ensure async client is created from pending config if needed.

        Raises:
            ValkeyVectorStoreError: If async client cannot be created.

        """
        if self._valkey_client_async:
            # Already have async client
            return

        if self._pending_async_config:
            # Create async client from pending config
            try:
                self._valkey_client_async = await GlideClient.create(
                    self._pending_async_config
                )
                logger.info("Created async client from URL configuration")
                # Clear pending config after creating client
                self._pending_async_config = None
            except Exception as e:
                raise ValkeyVectorStoreError(
                    f"Failed to create async client: {e}"
                ) from e
        else:
            # No way to create async client
            raise ValkeyVectorStoreError(
                "No async client available. Either provide valkey_client_async, "
                "valkey_url, or both valkey_client and valkey_client_async."
            )

    def _drop_index(self) -> None:
        """
        Drop the index synchronously.

        Raises:
            ValkeyVectorStoreError: If dropping the index fails.

        """
        try:
            sync_ft.dropindex(self._valkey_client, self.index_name)
            logger.info(f"Dropped index {self.index_name}")
        except Exception as e:
            raise ValkeyVectorStoreError(
                f"Failed to drop index {self.index_name}: {e}"
            ) from e

    async def _async_drop_index(self) -> None:
        """
        Drop the index asynchronously.

        Raises:
            ValkeyVectorStoreError: If dropping the index fails.

        """
        try:
            await ft.dropindex(self._valkey_client_async, self.index_name)
            logger.info(f"Dropped index {self.index_name}")
        except Exception as e:
            raise ValkeyVectorStoreError(
                f"Failed to drop index {self.index_name}: {e}"
            ) from e

    @property
    def index_name(self) -> str:
        """Return the name of the index based on the schema."""
        return self._schema.index.name

    @property
    def schema(self) -> ValkeyVectorStoreSchema:
        """Return the index schema."""
        return self._schema

    def set_return_fields(self, return_fields: List[str]) -> None:
        """Update the return fields for the query response."""
        self._return_fields = return_fields

    def index_exists(self) -> bool:
        """
        Check whether the index exists in Valkey.

        Returns:
            bool: True or False.

        """
        self._ensure_sync_client()
        try:
            sync_ft.info(self._valkey_client, self.index_name)
            return True
        except Exception as e:
            logger.debug(f"Valkey index check failed for {self.index_name}, {e}")
            return False

    async def async_index_exists(self) -> bool:
        """
        Check whether the index exists in Valkey.

        Returns:
            bool: True or False.

        """
        await self._ensure_async_client()
        try:
            await ft.info(self._valkey_client_async, self.index_name)
            return True
        except Exception as e:
            logger.debug(f"Valkey index check failed for {self.index_name}, {e}")
            return False

    def create_index(self, overwrite: Optional[bool] = None) -> None:
        """Create an index in Valkey."""
        if overwrite is None:
            overwrite = self._overwrite

        self._ensure_sync_client()

        try:
            exists = self.index_exists()

            if exists and overwrite:
                # Drop existing index
                self._drop_index()
            elif exists and not overwrite:
                logger.info(
                    f"Index {self.index_name} already exists, skipping creation"
                )
                return

            # Use schema fields directly with prefix
            options = FtCreateOptions(
                self._schema.index.storage_type,
                prefixes=[self._schema.index.prefix + self._schema.index.key_separator],
            )
            result = sync_ft.create(
                self._valkey_client, self.index_name, self._schema.fields, options
            )
            if result not in (b"OK", "OK"):
                raise ValkeyVectorStoreError(
                    f"FT.CREATE failed for index '{self.index_name}': {result!r}"
                )

            logger.info(f"Created index {self.index_name}")

        except ValkeyVectorStoreError:
            raise
        except Exception as e:
            raise ValkeyVectorStoreError(
                f"Failed to create index {self.index_name}: {e}"
            ) from e

    async def async_create_index(self, overwrite: Optional[bool] = None) -> None:
        """Create an async index in Valkey."""
        if overwrite is None:
            overwrite = self._overwrite

        await self._ensure_async_client()

        try:
            # Check if index exists without calling async_index_exists to avoid circular dependency
            exists = False
            try:
                await ft.info(self._valkey_client_async, self.index_name)
                exists = True
            except Exception:
                exists = False

            if exists and overwrite:
                # Drop existing index
                await self._async_drop_index()
            elif exists and not overwrite:
                logger.info(
                    f"Index {self.index_name} already exists, skipping creation"
                )
                self.created_async_index = True
                return

            # Use schema fields directly with prefix
            options = FtCreateOptions(
                self._schema.index.storage_type,
                prefixes=[self._schema.index.prefix + self._schema.index.key_separator],
            )
            logger.info(
                f"Creating index {self.index_name} with {len(self._schema.fields)} fields"
            )
            result = await ft.create(
                self._valkey_client_async, self.index_name, self._schema.fields, options
            )
            if result not in (b"OK", "OK"):
                raise ValkeyVectorStoreError(
                    f"FT.CREATE failed for index '{self.index_name}': {result!r}"
                )

            self.created_async_index = True
            logger.info(f"Created index {self.index_name}")

        except ValkeyVectorStoreError:
            raise
        except Exception as e:
            raise ValkeyVectorStoreError(
                f"Failed to create index {self.index_name}: {e}"
            ) from e

    def delete_index(self) -> None:
        """Delete the index and all documents."""
        self._ensure_sync_client()
        self._drop_index()

    async def async_delete_index(self) -> None:
        """Delete the index and all documents asynchronously."""
        await self._ensure_async_client()
        await self._async_drop_index()

    def _prepare_node_data(self, nodes: List[BaseNode]) -> List[tuple[str, dict, str]]:
        """
        Prepare node data for insertion.

        Returns:
            List of (key, fields, node_id) tuples

        """
        if len(nodes) == 0:
            return []

        # Get vector dimensions from schema
        vector_field = next(
            (
                f
                for f in self._schema.fields
                if hasattr(f, "name") and f.name == VECTOR_FIELD_NAME
            ),
            None,
        )
        if vector_field and hasattr(vector_field, "attributes"):
            expected_dims = vector_field.attributes.dimensions
        else:
            logger.warning(
                f"Vector field '{VECTOR_FIELD_NAME}' not found in schema or missing attributes. "
                f"Defaulting to {DEFAULT_EMBEDDING_DIM} dimensions. This may cause issues if your embeddings have different dimensions."
            )
            expected_dims = DEFAULT_EMBEDDING_DIM

        # Check embedding dimensions
        for node in nodes:
            embedding_len = len(node.get_embedding())
            if expected_dims != embedding_len:
                raise ValueError(
                    f"Attempting to index embeddings of dim {embedding_len} "
                    f"which doesn't match the index schema expectation of {expected_dims}."
                )

        # Prepare all nodes
        prepared_nodes = []
        for node in nodes:
            embedding = node.get_embedding()
            node_id = node.node_id
            key = f"{self._schema.index.prefix}{self._schema.index.key_separator}{node_id}"

            # Prepare hash fields
            fields = {
                NODE_ID_FIELD_NAME: node_id,
                DOC_ID_FIELD_NAME: node.ref_doc_id or "",
                TEXT_FIELD_NAME: node.get_content(metadata_mode=MetadataMode.NONE),
                VECTOR_FIELD_NAME: array_to_buffer(embedding, dtype="FLOAT32"),
            }

            # Add node content as JSON
            additional_metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )
            fields[NODE_CONTENT_FIELD_NAME] = json.dumps(additional_metadata)

            # Add additional metadata fields
            for meta_key, meta_value in additional_metadata.items():
                if meta_key not in fields and meta_key != "sub_dicts":
                    fields[meta_key] = str(meta_value) if meta_value is not None else ""

            prepared_nodes.append((key, fields, node_id))

        return prepared_nodes

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Add nodes to the index.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings

        Returns:
            List[str]: List of ids of the documents added to the index.

        """
        self._ensure_sync_client()

        prepared_nodes = self._prepare_node_data(nodes)
        node_ids = []

        batch = SyncBatch(True)
        for key, fields, node_id in prepared_nodes:
            batch.hset(key, fields)
            node_ids.append(node_id)
        try:
            self._valkey_client.exec(batch, raise_on_error=True)
            logger.info(f"Added {len(node_ids)} documents to index {self.index_name}")
            return node_ids
        except Exception as e:
            raise ValkeyVectorStoreError(
                f"Failed to add {len(node_ids)} nodes to index {self.index_name}: {e}"
            ) from e

    async def async_add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Add nodes to the index asynchronously.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings

        Returns:
            List[str]: List of ids of the documents added to the index.

        """
        await self._ensure_async_client()

        prepared_nodes = self._prepare_node_data(nodes)
        node_ids = []

        batch = ClusterBatch(True) if self._is_cluster else Batch(True)
        for key, fields, node_id in prepared_nodes:
            batch.hset(key, fields)
            node_ids.append(node_id)

        try:
            await self._valkey_client_async.exec(batch, raise_on_error=True)
            logger.info(f"Added {len(node_ids)} documents to index {self.index_name}")
            return node_ids
        except Exception as e:
            raise ValkeyVectorStoreError(
                f"Failed to add {len(node_ids)} nodes to index {self.index_name}: {e}"
            ) from e

    def delete_nodes(self, node_ids: List[str]) -> None:
        """Delete specific nodes by node_id."""
        self._ensure_sync_client()

        keys = [
            f"{self._schema.index.prefix}{self._schema.index.key_separator}{node_id}"
            for node_id in node_ids
        ]
        try:
            result = self._valkey_client.delete(keys)
            if result < len(node_ids):
                logger.warning(
                    f"Some nodes not found. Expected {len(node_ids)}, deleted {result}"
                )
        except Exception as e:
            raise ValkeyVectorStoreError(
                f"Failed to delete nodes from index {self.index_name}: {e}"
            ) from e

    async def async_delete_nodes(self, node_ids: List[str]) -> None:
        """Delete specific nodes by node_id asynchronously."""
        await self._ensure_async_client()

        keys = [
            f"{self._schema.index.prefix}{self._schema.index.key_separator}{node_id}"
            for node_id in node_ids
        ]
        try:
            result = await self._valkey_client_async.delete(keys)
            if result < len(node_ids):
                logger.warning(
                    f"Some nodes not found. Expected {len(node_ids)}, deleted {result}"
                )
        except Exception as e:
            raise ValkeyVectorStoreError(
                f"Failed to delete nodes from index {self.index_name}: {e}"
            ) from e

    def delete(self, ref_doc_id: str) -> None:
        """
        Delete nodes using the ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        self._ensure_sync_client()

        if not self.index_exists():
            logger.warning(f"Index {self.index_name} does not exist, cannot delete")
            return

        # Search for all nodes with this doc_id
        # For TAG fields, use the value directly without escaping (TAG fields are exact match)
        query = f"@{DOC_ID_FIELD_NAME}:{{{ref_doc_id}}}"
        options = FtSearchOptions(
            limit=FtSearchLimit(offset=0, count=10000),
            return_fields=[ReturnField(NODE_ID_FIELD_NAME)],
        )

        try:
            result = sync_ft.search(
                self._valkey_client, self.index_name, query, options
            )
            if isinstance(result, list) and len(result) > 0:
                count = result[0]
                if count == 0:
                    logger.info(f"No documents found with doc_id {ref_doc_id}")
                    return

                # Extract keys and delete (valkey-glide always returns dict format)
                node_keys = []
                if len(result) > 1:
                    for key in result[1]:
                        if isinstance(key, bytes):
                            key = key.decode()
                        node_keys.append(key)

                if node_keys:
                    result = self._valkey_client.delete(node_keys)
                    logger.info(f"Deleted {result} documents with doc_id {ref_doc_id}")
        except Exception as e:
            raise ValkeyVectorStoreError(
                f"Failed to delete documents with doc_id {ref_doc_id} from index {self.index_name}: {e}"
            ) from e

    async def async_delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using the ref_doc_id asynchronously.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        await self._ensure_async_client()
        await self.async_index_exists()

        # Search for all nodes with this doc_id
        query = f"@{DOC_ID_FIELD_NAME}:{{{ref_doc_id}}}"
        options = FtSearchOptions(
            limit=FtSearchLimit(offset=0, count=10000),
            return_fields=[
                ReturnField(NODE_ID_FIELD_NAME),
                ReturnField(DOC_ID_FIELD_NAME),
            ],
        )

        try:
            result = await ft.search(
                self._valkey_client_async, self.index_name, query, options
            )

            if isinstance(result, list) and len(result) > 0:
                count = result[0]

                # Extract keys and delete (valkey-glide always returns dict format)
                node_keys = []
                if len(result) > 1:
                    for key in result[1]:
                        if isinstance(key, bytes):
                            key = key.decode()
                        node_keys.append(key)

                if node_keys:
                    result = await self._valkey_client_async.delete(node_keys)
                    logger.info(f"Deleted {result} documents with doc_id {ref_doc_id}")
                else:
                    logger.debug(f"No keys found to delete for doc_id {ref_doc_id}")
            else:
                logger.debug(f"No results or invalid result format")
        except Exception as e:
            raise ValkeyVectorStoreError(
                f"Failed to delete documents with doc_id {ref_doc_id} from index {self.index_name}: {e}"
            ) from e

    # Aliases for compatibility
    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Alias for async_delete."""
        await self.async_delete(ref_doc_id, **delete_kwargs)

    async def adelete_nodes(self, node_ids: List[str], **delete_kwargs: Any) -> None:
        """Alias for async_delete_nodes."""
        await self.async_delete_nodes(node_ids)

    def _build_query_and_options(
        self, query: VectorStoreQuery
    ) -> tuple[str, FtSearchOptions]:
        """
        Build query string and search options.

        Returns:
            Tuple of (query_string, search_options)

        """
        if query.query_embedding is None and not query.filters:
            raise ValueError(
                "Either query_embedding or metadata filters are required for querying."
            )

        if query.query_embedding is not None:
            # Vector search
            query_vec_bytes = array_to_buffer(query.query_embedding)

            # Build KNN query with optional filters
            if query.filters:
                filter_str = self._build_filter_string(query.filters)
                knn_query = f"({filter_str})=>[KNN {query.similarity_top_k} @{VECTOR_FIELD_NAME} $query_vector]"
            else:
                knn_query = f"*=>[KNN {query.similarity_top_k} @{VECTOR_FIELD_NAME} $query_vector]"

            # Build search options
            options = FtSearchOptions(
                limit=FtSearchLimit(offset=0, count=query.similarity_top_k),
                params={"query_vector": query_vec_bytes},
            )

            logger.info(
                f"Executing KNN query on index {self.index_name}, top_k={query.similarity_top_k}"
            )
            logger.debug(f"Query: {knn_query}")
        else:
            # Filter-only query
            filter_str = self._build_filter_string(query.filters)
            knn_query = filter_str

            options = FtSearchOptions(
                limit=FtSearchLimit(offset=0, count=query.similarity_top_k),
            )

            if self._return_fields:
                options.return_fields = [
                    ReturnField(field) for field in self._return_fields
                ]

            logger.info(
                f"Executing filter query on index {self.index_name}: {filter_str}"
            )

        return knn_query, options

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query the index.

        Args:
            query (VectorStoreQuery): query object

        Returns:
            VectorStoreQueryResult: query result

        """
        self._ensure_sync_client()
        self.index_exists()

        knn_query, options = self._build_query_and_options(query)

        try:
            result = sync_ft.search(
                self._valkey_client, self.index_name, knn_query, options
            )
            logger.info(
                f"Query returned {result[0] if isinstance(result, list) and len(result) > 0 else 0} results"
            )
            logger.debug(f"Raw result: {result}")
            return self._process_search_results(
                result, query.query_embedding is not None
            )
        except Exception as e:
            raise ValkeyVectorStoreError(
                f"Failed to query index {self.index_name}: {e}"
            ) from e

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Query the index asynchronously.

        Args:
            query (VectorStoreQuery): query object

        Returns:
            VectorStoreQueryResult: query result

        """
        await self._ensure_async_client()

        knn_query, options = self._build_query_and_options(query)

        try:
            result = await ft.search(
                self._valkey_client_async, self.index_name, knn_query, options
            )
            logger.info(
                f"Query returned {result[0] if isinstance(result, list) and len(result) > 0 else 0} results"
            )
            logger.debug(f"Raw result: {result}")
            return self._process_search_results(
                result, query.query_embedding is not None
            )
        except Exception as e:
            raise ValkeyVectorStoreError(
                f"Failed to query index {self.index_name}: {e}"
            ) from e

    def _build_filter_string(self, filters: Optional[MetadataFilters]) -> str:
        """Build filter string for FT.SEARCH."""
        if not filters or not filters.filters:
            return "*"

        filter_parts = []
        for f in filters.filters:
            if isinstance(f, MetadataFilter):
                field = f.key
                value = f.value
                op = f.operator

                if op == FilterOperator.EQ:
                    filter_parts.append(f"@{field}:{{{value}}}")
                elif op == FilterOperator.NE:
                    filter_parts.append(f"-@{field}:{{{value}}}")
                elif op == FilterOperator.GT:
                    filter_parts.append(f"@{field}:[({value} +inf]")
                elif op == FilterOperator.LT:
                    filter_parts.append(f"@{field}:[-inf ({value}]")
                elif op == FilterOperator.GTE:
                    filter_parts.append(f"@{field}:[{value} +inf]")
                elif op == FilterOperator.LTE:
                    filter_parts.append(f"@{field}:[-inf {value}]")

        if not filter_parts:
            return "*"

        # Combine with AND
        return " ".join(filter_parts)

    def _extract_node_and_score(self, doc: Dict[str, Any], has_vector: bool) -> tuple:
        """
        Extract a node and (optional) score from a document.

        Args:
            doc: Document dictionary from search results
            has_vector: Whether this was a vector search (to extract score)

        Returns:
            Tuple of (node, score) where score is None for filter-only queries

        """
        try:
            node_content = doc.get(NODE_CONTENT_FIELD_NAME)
            # Parse if it's a string
            if isinstance(node_content, str):
                node_content = json.loads(node_content)

            # node_content is a dict with metadata fields + nested _node_content
            # Extract the nested _node_content which has the full node serialization
            if (
                isinstance(node_content, dict)
                and NODE_CONTENT_FIELD_NAME in node_content
            ):
                # Use the nested _node_content value
                inner_content = node_content[NODE_CONTENT_FIELD_NAME]
                node = metadata_dict_to_node({NODE_CONTENT_FIELD_NAME: inner_content})
            else:
                # Fallback: use the content directly
                node = metadata_dict_to_node({NODE_CONTENT_FIELD_NAME: node_content})

            node.text = doc.get(TEXT_FIELD_NAME, node.get_content())
        except Exception:
            # Fallback: create basic node
            node = TextNode(
                text=doc.get(TEXT_FIELD_NAME, ""),
                id_=doc.get(NODE_ID_FIELD_NAME),
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(
                        node_id=doc.get(DOC_ID_FIELD_NAME, "")
                    )
                },
            )

        score = None
        if has_vector:
            # Look for score field - Valkey returns it as __vector_score
            score_field = "__vector_score"
            if score_field in doc:
                try:
                    distance = float(doc[score_field])
                    # Convert distance to similarity (1 - distance for cosine)
                    score = 1.0 - distance
                except (ValueError, TypeError):
                    pass

        return node, score

    def _process_search_results(
        self, result: Any, has_vector: bool
    ) -> VectorStoreQueryResult:
        """Process FT.SEARCH results."""
        nodes = []
        ids = []
        similarities = []

        if not isinstance(result, list) or len(result) == 0:
            logger.warning(f"Empty or invalid result from FT.SEARCH: {type(result)}")
            return VectorStoreQueryResult(
                nodes=nodes, ids=ids, similarities=similarities
            )

        count = result[0]
        logger.info(f"Processing {count} search results")

        if len(result) > 1:
            results_dict = result[1]

            for key, doc in results_dict.items():
                if isinstance(key, bytes):
                    key = key.decode()

                # Convert bytes to strings in doc
                doc_dict = {}
                for field_name, field_value in doc.items():
                    if isinstance(field_name, bytes):
                        field_name = field_name.decode()
                    # Skip decoding for vector field (binary data)
                    if field_name == VECTOR_FIELD_NAME:
                        doc_dict[field_name] = field_value
                    elif isinstance(field_value, bytes):
                        try:
                            field_value = field_value.decode()
                        except UnicodeDecodeError:
                            # Keep as bytes if can't decode
                            pass
                        doc_dict[field_name] = field_value
                    else:
                        doc_dict[field_name] = field_value

                logger.debug(
                    f"Processing document with key {key}: {list(doc_dict.keys())}"
                )

                try:
                    node, score = self._extract_node_and_score(doc_dict, has_vector)
                    nodes.append(node)
                    ids.append(doc_dict.get(NODE_ID_FIELD_NAME, ""))
                    if score is not None:
                        similarities.append(score)

                    logger.debug(
                        f"Successfully parsed node {doc_dict.get(NODE_ID_FIELD_NAME)}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to parse node: {e}, fields: {list(doc_dict.keys())}"
                    )
                    logger.warning(f"Traceback: {traceback.format_exc()}")
                    continue

        logger.info(f"Found {len(nodes)} results with ids {ids}")
        return VectorStoreQueryResult(
            nodes=nodes, ids=ids, similarities=similarities if similarities else None
        )

    def persist(
        self,
        persist_path: Optional[str] = None,
        in_background: bool = True,
    ) -> None:
        """
        Persist the vector store to disk.

        For Valkey, more notes here: https://valkey.io/topics/persistence/

        Args:
            persist_path (str): Path to persist the vector store to. (doesn't apply)
            in_background (bool, optional): Persist in background. Defaults to True.

        """
        self._ensure_sync_client()

        try:
            if in_background:
                logger.info("Saving index to disk in background")
                self._valkey_client.custom_command(["BGSAVE"])
            else:
                logger.info("Saving index to disk")
                self._valkey_client.custom_command(["SAVE"])
        except Exception as e:
            raise ValkeyVectorStoreError(f"Failed to persist index to disk: {e}") from e

    async def apersist(
        self,
        persist_path: Optional[str] = None,
        in_background: bool = True,
    ) -> None:
        """
        Persist the vector store to disk asynchronously.

        Args:
            persist_path (str): Path to persist the vector store to. (doesn't apply)
            in_background (bool, optional): Persist in background. Defaults to True.

        """
        await self._ensure_async_client()

        try:
            if in_background:
                logger.info("Saving index to disk in background")
                await self._valkey_client_async.custom_command(["BGSAVE"])
            else:
                logger.info("Saving index to disk")
                await self._valkey_client_async.custom_command(["SAVE"])
        except Exception as e:
            raise ValkeyVectorStoreError(f"Failed to persist index to disk: {e}") from e
