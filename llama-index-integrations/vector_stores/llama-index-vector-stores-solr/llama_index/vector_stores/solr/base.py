"""Apache Solr vector store."""

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Annotated, Any, ClassVar, Optional, Union

from annotated_types import MinLen
from pydantic import (
    ConfigDict,
    Field,
    SkipValidation,
    field_validator,
)

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.solr.constants import (
    ESCAPE_RULES_NESTED_LUCENE_DISMAX,
)
from llama_index.vector_stores.solr.query_utils import (
    escape_query_characters,
    recursively_unpack_filters,
)
from llama_index.vector_stores.solr.types import BoostedTextField, SolrQueryDict

logger = logging.getLogger(__name__)


class ApacheSolrVectorStore(BasePydanticVectorStore):
    """
    A LlamaIndex vector store implementation for Apache Solr.

    This vector store provides integration with Apache Solr, supporting
    both dense vector similarity search (KNN) and sparse text search (BM25).

    Key Features:

    * Dense vector embeddings with KNN similarity search
    * Sparse text search with BM25 scoring and field boosting
    * Metadata filtering with various operators
    * Async/sync operations
    * Automatic query escaping and field preprocessing

    Field Mapping: the vector store maps LlamaIndex node attributes
    to Solr fields:

    * ``nodeid_field``: Maps to ``node.id_`` (required)
    * ``content_field``: Maps to ``node.get_content()`` (optional)
    * ``embedding_field``: Maps to ``node.get_embedding()`` (optional)
    * ``docid_field``: Maps to ``node.ref_doc_id`` (optional)
    * ``metadata fields``: Mapped via ``metadata_to_solr_field_mapping``

    Query Modes:

    * ``DEFAULT``: Dense vector KNN search using embeddings
    * ``TEXT_SEARCH``: Sparse BM25 text search with field boosting
    """

    # Core client properties
    sync_client: SkipValidation[Any] = Field(
        ...,
        exclude=True,
        description="Synchronous Solr client instance for blocking operations.",
    )
    async_client: SkipValidation[Any] = Field(
        ...,
        exclude=True,
        description="Asynchronous Solr client instance for non-blocking operations.",
    )

    # Essential field mappings
    nodeid_field: str = Field(
        ...,
        description=(
            "Solr field name that uniquely identifies a node (required). Must be unique across all documents and maps to the LlamaIndex `node.id_`."
        ),
    )
    docid_field: Optional[str] = Field(
        default=None,
        description=(
            "Solr field name for the document ID (optional). Maps to `node.ref_doc_id` and is required for document-level operations like deletion."
        ),
    )
    content_field: Optional[str] = Field(
        default=None,
        description=(
            "Solr field name for storing the node's text content (optional). Maps to `node.get_content()`; required for BM25 / text search."
        ),
    )
    embedding_field: Optional[str] = Field(
        default=None,
        description=(
            "Solr field name for storing embedding vectors (optional). Maps to `node.get_embedding()`; required for vector similarity (KNN) search."
        ),
    )
    metadata_to_solr_field_mapping: Optional[list[tuple[str, str]]] = Field(
        default=None,
        description=(
            "Mapping from node metadata keys to Solr field names (optional). Each tuple is (metadata_key, solr_field). Enables structured metadata filtering."
        ),
    )

    # Configuration options
    text_search_fields: Optional[Annotated[Sequence[BoostedTextField], MinLen(1)]] = (
        Field(
            default=None,
            description=(
                "Fields used for BM25 text search with optional boosting. Sequence of BoostedTextField; required for TEXT_SEARCH mode."
            ),
        )
    )
    output_fields: Annotated[Sequence[str], MinLen(1)] = Field(
        default=["*", "score"],
        description=(
            "Default fields to return in query results. Include 'score' automatically for relevance; use '*' for all stored fields or list specific ones."
        ),
    )

    # Serialization configuration
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, frozen=True
    )

    # Required for LlamaIndex API compatibility
    stores_text: bool = True
    stores_node: bool = True
    flat_metadata: bool = False

    @field_validator("output_fields")
    @classmethod
    def _validate_output_fields(cls, value: Sequence[str]) -> list[str]:
        """
        Ensure 'score' field is always included in output_fields during initialization.

        Args:
            value (Sequence[str]): The original output fields
        Returns:
            Modified output fields with 'score' always included

        """
        result = list(value)
        if "score" not in result:
            result.append("score")
        return result

    @field_validator("text_search_fields", mode="before")
    def _validate_text_search_fields(
        cls, v: Optional[list[Union[str, BoostedTextField]]]
    ) -> Optional[list[BoostedTextField]]:
        """Validate and convert text search fields to BoostedTextField instances."""
        if v is None:
            return None

        def to_boosted(item: Union[str, BoostedTextField]) -> BoostedTextField:
            if isinstance(item, str):
                return BoostedTextField(field=item)
            return item

        return [to_boosted(item) for item in v]

    @property
    def client(self) -> Any:
        """Return synchronous Solr client."""
        return self.sync_client

    @property
    def aclient(self) -> Any:
        """Return asynchronous Solr client."""
        return self.async_client

    def _build_dense_query(
        self, query: VectorStoreQuery, solr_query: SolrQueryDict
    ) -> SolrQueryDict:
        """
        Build a dense vector KNN query for Solr.

        Args:
            query: The vector store query containing embedding and parameters
            solr_query: The base Solr query dictionary to build upon
        Returns:
            Updated Solr query dictionary with dense vector search parameters
        Raises:
            ValueError: If no embedding field is specified in either query or vector store

        """
        if query.embedding_field is not None:
            embedding_field = query.embedding_field
            logger.debug("Using embedding field from query: %s", embedding_field)

        elif self.embedding_field is not None:
            embedding_field = self.embedding_field
            logger.debug("Using embedding field from vector store: %s", embedding_field)

        else:
            raise ValueError(
                "No embedding field name specified in query or vector store. "
                "Either set 'embedding_field' on the VectorStoreQuery or configure "
                "'embedding_field' when initializing ApacheSolrVectorStore"
            )

        if query.query_embedding is None:
            logger.warning(
                "`query.query_embedding` is None, retrieval results will not be meaningful."
            )

        solr_query["q"] = (
            f"{{!knn f={embedding_field} topK={query.similarity_top_k}}}{query.query_embedding}"
        )
        rows_value = None or query.similarity_top_k
        solr_query["rows"] = str(rows_value)
        return solr_query

    def _build_bm25_query(
        self, query: VectorStoreQuery, solr_query: SolrQueryDict
    ) -> SolrQueryDict:
        """
        Build a BM25 text search query for Solr.

        Args:
            query: The vector store query containing the query string and parameters
            solr_query: The base Solr query dictionary to build upon
        Returns:
            Updated Solr query dictionary with BM25 search parameters
        Raises:
            ValueError: If no text search fields are available or query string is None

        """
        if query.query_str is None:
            raise ValueError("Query string cannot be None for BM25 search")

        # Use text_search_fields from the vector store
        if self.text_search_fields is None:
            raise ValueError(
                "text_search_fields must be specified in the vector store config for BM25 search"
            )

        user_query = escape_query_characters(
            query.query_str, translation_table=ESCAPE_RULES_NESTED_LUCENE_DISMAX
        )

        # Join the search fields with spaces for the Solr qf parameter
        search_fields_str = " ".join(
            [
                text_search_field.get_query_str()
                for text_search_field in self.text_search_fields
            ]
        )
        solr_query["q"] = (
            f"{{!dismax deftype=lucene, qf='{search_fields_str}' v='{user_query}'}}"
        )
        # Use rows from query if provided, otherwise fall back to similarity_top_k
        rows_value = None or query.sparse_top_k
        solr_query["rows"] = str(rows_value)
        return solr_query

    def _to_solr_query(self, query: VectorStoreQuery) -> SolrQueryDict:
        """Generate a KNN Solr query."""
        solr_query: SolrQueryDict = {"q": "*:*", "fq": []}

        if (
            query.mode == VectorStoreQueryMode.DEFAULT
            and query.query_embedding is not None
        ):
            solr_query = self._build_dense_query(query, solr_query)

        elif query.mode == VectorStoreQueryMode.TEXT_SEARCH:
            solr_query = self._build_bm25_query(query, solr_query)

        if query.doc_ids is not None:
            if self.docid_field is None:
                raise ValueError(
                    "`docid_field` must be passed during initialization to filter on docid"
                )
            solr_query["fq"].append(
                f"{self.docid_field}:({' OR '.join(query.doc_ids)})"
            )
        if query.node_ids is not None and len(query.node_ids) > 0:
            solr_query["fq"].append(
                f"{self.nodeid_field}:({' OR '.join(query.node_ids)})"
            )
        if query.output_fields is not None:
            # Use output fields from query, ensuring score is always included
            output_fields = self._validate_output_fields(query.output_fields)
            solr_query["fl"] = ",".join(output_fields)
            logger.info("Using output fields from query: %s", output_fields)
        else:
            # Use default output fields from vector store, ensuring score is always included
            solr_query["fl"] = ",".join(self.output_fields)
            logger.info(
                "Using default output fields from vector store: %s", self.output_fields
            )

        if query.filters:
            filter_queries = recursively_unpack_filters(query.filters)
            solr_query["fq"].extend(filter_queries)

        logger.debug(
            "Converted input query into Solr query dictionary, input=%s, output=%s",
            query,
            solr_query,
        )
        return solr_query

    def _process_query_results(
        self, results: list[dict[str, Any]]
    ) -> VectorStoreQueryResult:
        """
        Convert Solr search results to LlamaIndex VectorStoreQueryResult format.
        This method transforms raw Solr documents into LlamaIndex TextNode objects
        and packages them with similarity scores and metadata into a structured
        query result. It handles field mapping, metadata extraction.

        Args:
            results: List of Solr document dictionaries from search response.
                Each dictionary contains field values as returned by Solr.

        Returns:
            A :py:class:`VectorStoreQueryResult` containing:
            * ``nodes``: List of :py:class:`TextNode` objects with content and metadata
            * ``ids``: List of node IDs corresponding to each node
            * ``similarities``: List of similarity scores (if available)

        Raises:
            ValueError: If the number of similarity scores doesn't match the
                number of nodes (partial scoring is not supported).

        Note:
            * Metadata fields are automatically identified by excluding known
              system fields (``nodeid_field``, ``content_field``, etc.)
            * The 'score' field from Solr is extracted as similarity scores
            * Missing optional fields (``content``, ``embedding``) are handled gracefully

        """
        ids, nodes, similarities = [], [], []
        for result in results:
            metadata_fields = result.keys() - {
                self.nodeid_field,
                self.content_field,
                self.embedding_field,
                self.docid_field,
                "score",
            }

            ids.append(result[self.nodeid_field])

            node = TextNode(
                id_=result[self.nodeid_field],
                # input must be a string, if missing use empty string
                text=result[self.content_field] if self.content_field else "",
                embedding=(
                    result[self.embedding_field] if self.embedding_field else None
                ),
                metadata={f: result[f] for f in metadata_fields},
            )
            nodes.append(node)
            if "score" in result:
                similarities.append(result["score"])

        if len(similarities) == 0:
            return VectorStoreQueryResult(nodes=nodes, ids=ids)
        elif 0 < len(similarities) < len(nodes):
            raise ValueError(
                "The number of similarities (scores) does not match the number of nodes"
            )
        else:
            return VectorStoreQueryResult(
                nodes=nodes, ids=ids, similarities=similarities
            )

    def _validate_query_mode(self, query: VectorStoreQuery) -> None:
        """
        Validate that the query mode is supported by this vector store.

        This method ensures that the requested query mode is compatible with
        the current Solr vector store implementation.

        Supported Modes:
        * ``DEFAULT``: Dense vector similarity search using KNN with embeddings
        * ``TEXT_SEARCH``: Sparse text search using BM25 with field boosting

        Args:
            query:
                The vector store query containing the mode to validate. The mode is
                checked against supported :py:class:`VectorStoreQueryMode` values.

        Raises:
            ValueError: If the query mode is not supported. Unsupported modes
                include any future modes not yet implemented in the Solr backend.

        Note:
            This validation occurs before query execution to provide clear
            error messages for unsupported operations. Future versions may
            support additional query modes like hybrid search.

        """
        if (
            query.mode == VectorStoreQueryMode.DEFAULT
            or query.mode == VectorStoreQueryMode.TEXT_SEARCH
        ):
            return
        else:
            raise ValueError(
                f"ApacheSolrVectorStore does not support {query.mode} yet."
            )

    def query(
        self, query: VectorStoreQuery, **search_kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Execute a synchronous search query against the Solr vector store.

        This method supports both dense vector similarity search (KNN) and sparse
        text search (BM25) depending on the query mode and parameters. It handles
        query validation, Solr query construction, execution, and result processing.

        Query Types:

        * Dense Vector Search: Uses ``query_embedding`` for KNN similarity search
        * Text Search: Uses ``query_str`` for BM25 text search with field boosting
        * Filtered Search: Combines vector/text search with metadata filters

        Supported Filter Operations:

        * ``EQ``, ``NE``: Equality and inequality comparisons
        * ``GT``, ``GTE``, ``LT``, ``LTE``: Numeric range comparisons
        * ``IN``, ``NIN``: List membership tests
        * ``TEXT_MATCH``: Exact text matching

        Unsupported Filter Operations:

        * ``ANY``, ``ALL``: Complex logical operations
        * ``TEXT_MATCH_INSENSITIVE``: Case-insensitive text matching
        * ``CONTAINS``: Substring matching

        Args:
            query:
                The vector store query containing search parameters:

                * ``query_embedding``: Dense vector for similarity search (DEFAULT mode)
                * ``query_str``: Text string for BM25 search (TEXT_SEARCH mode)
                * ``mode``: ``VectorStoreQueryMode`` (DEFAULT or TEXT_SEARCH)
                * ``similarity_top_k``: Number of results for vector search
                * ``sparse_top_k``: Number of results for text search
                * ``filters``: Optional metadata filters for constraining results
                * ``doc_ids``: Optional list of document IDs to filter by
                * ``node_ids``: Optional list of node IDs to filter by
                * ``output_fields``: Optional list of fields to return
            **search_kwargs: Extra keyword arguments (ignored for compatibility)

        Returns:
            VectorStoreQueryResult containing:

            * nodes: List of TextNode objects with content and metadata
            * ids: List of corresponding node IDs
            * similarities: List of similarity scores (when available)

        Raises:
            ValueError: If the query mode is unsupported, or if required fields
                are missing (e.g., ``embedding_field`` for vector search, ``docid_field``
                for document filtering)

        Note:
            This method performs synchronous I/O operations. For better performance
            in async contexts, use the :py:meth:`aquery` method instead.

        """
        del search_kwargs  # unused

        self._validate_query_mode(query)
        solr_query = self._to_solr_query(query)
        results = self.sync_client.search(solr_query)
        return self._process_query_results(results.response.docs)

    async def aquery(
        self, query: VectorStoreQuery, **search_kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Execute an asynchronous search query against the Solr vector store.

        This method supports both dense vector similarity search (KNN) and sparse
        text search (BM25) depending on the query mode and parameters. It handles
        query validation, Solr query construction, execution, and result processing.

        Query Types:

        * Dense Vector Search: Uses ``query_embedding`` for KNN similarity search
        * Text Search: Uses ``query_str`` for BM25 text search with field boosting
        * Filtered Search: Combines vector/text search with metadata filters

        Supported Filter Operations:

        * ``EQ``, ``NE``: Equality and inequality comparisons
        * ``GT``, ``GTE``, ``LT``, ``LTE``: Numeric range comparisons
        * ``IN``, ``NIN``: List membership tests
        * ``TEXT_MATCH``: Exact text matching

        Unsupported Filter Operations:

        * ``ANY``, ``ALL``: Complex logical operations
        * ``TEXT_MATCH_INSENSITIVE``: Case-insensitive text matching
        * ``CONTAINS``: Substring matching

        Args:
            query:
                The vector store query containing search parameters:

                * ``query_embedding``: Dense vector for similarity search (DEFAULT mode)
                * ``query_str``: Text string for BM25 search (TEXT_SEARCH mode)
                * ``mode``: ``VectorStoreQueryMode`` (DEFAULT or TEXT_SEARCH)
                * ``similarity_top_k``: Number of results for vector search
                * ``sparse_top_k``: Number of results for text search
                * ``filters``: Optional metadata filters for constraining results
                * ``doc_ids``: Optional list of document IDs to filter by
                * ``node_ids``: Optional list of node IDs to filter by
                * ``output_fields``: Optional list of fields to return
            **search_kwargs: Extra keyword arguments (ignored for compatibility)

        Returns:
            VectorStoreQueryResult containing:

            * nodes: List of TextNode objects with content and metadata
            * ids: List of corresponding node IDs
            * similarities: List of similarity scores (when available)

        Raises:
            ValueError: If the query mode is unsupported, or if required fields
                are missing (e.g., ``embedding_field`` for vector search, ``docid_field``
                for document filtering)

        """
        del search_kwargs  # unused

        self._validate_query_mode(query)
        solr_query = self._to_solr_query(query)
        results = await self.async_client.search(solr_query)
        return self._process_query_results(results.response.docs)

    def _get_data_from_node(self, node: BaseNode) -> dict[str, Any]:
        """
        Transform a LlamaIndex node into a Solr document dictionary.
        This method maps LlamaIndex node attributes to Solr fields based on the
        vector store configuration. It handles content extraction, embedding
        mapping, metadata processing.

        Args:
            node: LlamaIndex BaseNode containing content, metadata,
                to be stored in Solr.

        Returns:
            Dictionary representing a Solr document with mapped fields:
                - id: Always maps to node.node_id (required)
                - content_field: Maps to node.get_content() (if configured)
                - embedding_field: Maps to node.get_embedding() (if configured)
                - docid_field: Maps to node.ref_doc_id (if configured)
                - metadata fields: Mapped via metadata_to_solr_field_mapping

        Field Mapping Process:
            1. Always includes node ID as 'id' field
            2. Extracts content if content_field is configured
            3. Extracts embedding if embedding_field is configured
            4. Includes document ID if docid_field is configured
            5. Maps metadata using configured field mappings with preprocessing

        Note:
            This is an internal method used by add() and async_add() operations.
            The returned dictionary must be compatible with the Solr schema.

        """
        data: dict[str, Any] = {self.nodeid_field: node.node_id}
        if self.content_field is not None:
            data[self.content_field] = node.get_content()
        if self.embedding_field is not None:
            data[self.embedding_field] = node.get_embedding()
        if self.docid_field is not None:
            data[self.docid_field] = node.ref_doc_id
        if self.metadata_to_solr_field_mapping is not None:
            for metadata_key, solr_key in self.metadata_to_solr_field_mapping:
                if metadata_key in node.metadata:
                    data[solr_key] = node.metadata[metadata_key]
        return data

    def _get_data_from_nodes(
        self, nodes: Sequence[BaseNode]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        # helper to avoid double iteration, it gets expensive at large batch sizes
        logger.debug("Extracting data from %d nodes", len(nodes))
        data: list[dict[str, Any]] = []
        node_ids: list[str] = []
        for node in nodes:
            node_ids.append(node.id_)
            data.append(self._get_data_from_node(node))
        return node_ids, data

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:
        """
        Synchronously add nodes (documents) to a Solr collection.

        Mapping from Solr fields to :py:class:`llama_index.core.schema.BaseNode` attributes
        should be as follows:

        * ``nodeid_field`` -> ``node_id``
        * ``content_field`` -> ``content``
        * ``embedding_field`` -> ``embedding``
        * ``docid_field`` -> ``ref_doc_id``

        All other fields corresponding to the Solr collection should be packed as a single
        ``dict`` in the ``metadata`` field.

        Args:
            nodes: The nodes (documents) to be added to the Solr collection.
            **add_kwargs:
                Extra keyword arguments.

        Returns:
            A list of node IDs for each node added to the store.

        """
        del add_kwargs  # unused

        if not nodes:
            raise ValueError("Call to 'add' with no contents")

        start = time.perf_counter()
        node_ids, data = self._get_data_from_nodes(nodes)
        self.sync_client.add(data)
        logger.info(
            "Added %d documents to Solr in %0.2f seconds",
            len(data),
            time.perf_counter() - start,
        )
        return node_ids

    async def async_add(
        self,
        nodes: Sequence[BaseNode],
        **add_kwargs: Any,
    ) -> list[str]:
        """
        Asynchronously add nodes (documents) to a Solr collection.

        Mapping from Solr fields to :py:class:`llama_index.core.schema.BaseNode` attributes
        should be as follows:

        * ``nodeid_field`` -> ``node_id``
        * ``content_field`` -> ``content``
        * ``embedding_field`` -> ``embedding``
        * ``docid_field`` -> ``ref_doc_id``

        All other fields corresponding to the Solr collection should be packed as a single
        ``dict`` in the ``metadata`` field.

        Args:
            nodes: The nodes (documents) to be added to the Solr collection.
            **add_kwargs:
                Extra keyword arguments.

        Returns:
            A list of node IDs for each node added to the store.

        Raises:
            ValueError: If called with an empty list of nodes.

        """
        del add_kwargs  # unused

        if not nodes:
            raise ValueError("Call to 'async_add' with no contents")

        start = time.perf_counter()
        node_ids, data = self._get_data_from_nodes(nodes)
        await self.async_client.add(data)
        logger.info(
            "Added %d documents to Solr in %0.2f seconds",
            len(data),
            time.perf_counter() - start,
        )
        return node_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Synchronously delete a node from the collection using its reference document ID.

        Args:
            ref_doc_id: The reference document ID of the node to be deleted.
            **delete_kwargs:
                Extra keyword arguments, ignored by this implementation. These are added
                solely for interface compatibility.

        Raises:
            ValueError:
                If a ``docid_field`` was not passed to this vector store at
                initialization.

        """
        del delete_kwargs  # unused

        logger.debug("Deleting documents from Solr using query: %s", ref_doc_id)
        self.sync_client.delete_by_id([ref_doc_id])

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Asynchronously delete a node from the collection using its reference document ID.

        Args:
            ref_doc_id: The reference document ID of the node to be deleted.
            **delete_kwargs:
                Extra keyword arguments, ignored by this implementation. These are added
                solely for interface compatibility.

        Raises:
            ValueError:
                If a ``docid_field`` was not passed to this vector store at
                initialization.

        """
        del delete_kwargs  # unused

        logger.debug("Deleting documents from Solr using query: %s", ref_doc_id)
        await self.async_client.delete_by_id([ref_doc_id])

    def _build_delete_nodes_query(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> str:
        if not node_ids and not filters:
            raise ValueError(
                "At least one of `node_ids` or `filters` must be passed to `delete_nodes`"
            )

        queries: list[str] = []
        if node_ids:
            queries.append(f"{self.nodeid_field}:({' OR '.join(node_ids)})")
        if filters is not None:
            queries.extend(recursively_unpack_filters(filters))

        if not queries:
            raise ValueError(
                "Neither `node_ids` nor non-empty `filters` were passed to `delete_nodes`"
            )
        elif len(queries) == 1:
            return queries[0]
        return f"({' AND '.join(q for q in queries if q)})"

    def delete_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Synchronously delete nodes from vector store based on node ids.

        Args:
            node_ids: The node IDs to delete.
            filters: The filters to be applied to the node when deleting.
            **delete_kwargs:
                Extra keyword arguments, ignored by this implementation. These are added
                solely for interface compatibility.

        """
        del delete_kwargs  # unused

        has_filters = filters is not None and len(filters.filters) > 0
        # we can efficiently delete by ID if no filters are specified

        if node_ids and not has_filters:
            logger.debug("Deleting %d nodes from Solr by ID", len(node_ids))
            self.sync_client.delete_by_id(node_ids)

        # otherwise, build a query to delete by IDs+filters
        else:
            query_string = self._build_delete_nodes_query(node_ids, filters)
            logger.debug(
                "Deleting nodes from Solr using query: %s", query_string
            )  # pragma: no cover
            self.sync_client.delete_by_query(query_string)

    async def adelete_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Asynchronously delete nodes from vector store based on node ids.

        Args:
            node_ids: The node IDs to delete.
            filters: The filters to be applied to the node when deleting.
            **delete_kwargs:
                Extra keyword arguments, ignored by this implementation. These are added
                solely for interface compatibility.

        """
        del delete_kwargs  # unused

        has_filters = filters is not None and len(filters.filters) > 0
        # we can efficiently delete by ID if no filters are specified
        if node_ids and not has_filters:
            logger.debug("Deleting %d nodes from Solr by ID", len(node_ids))
            await self.async_client.delete_by_id(node_ids)

        # otherwise, build a query to delete by IDs+filters
        else:
            query_string = self._build_delete_nodes_query(node_ids, filters)
            logger.debug("Deleting nodes from Solr using query: %s", query_string)
            await self.async_client.delete_by_query(query_string)

    def clear(self) -> None:
        """
        Delete all documents from the Solr collection synchronously.
        This action is not reversible!
        """
        self.sync_client.clear_collection()

    async def aclear(self) -> None:
        """
        Delete all documents from the Solr collection asynchronously.
        This action is not reversible!
        """
        await self.async_client.clear_collection()

    def close(self) -> None:
        """Close the Solr client synchronously."""
        self.sync_client.close()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop: create a temporary loop and close cleanly
            asyncio.run(self.async_client.close())
        else:
            # Running loop: schedule async close (not awaited)
            loop.create_task(self.async_client.close())  # noqa: RUF006

    async def aclose(self) -> None:
        """Explicit aclose for callers running inside an event loop."""
        self.sync_client.close()
        await self.async_client.close()

    def __del__(self) -> None:
        """
        Clean up the client for shutdown.
        This action is not reversible, and should only be called one time.
        """
        try:
            self.close()
        except RuntimeError as exc:
            logger.debug(
                "No running event loop, nothing to close, type=%s err='%s'",
                type(exc),
                exc,
            )
        except Exception as exc:
            logger.warning(
                "Failed to close the async Solr client, type=%s err='%s'",
                type(exc),
                exc,
            )
